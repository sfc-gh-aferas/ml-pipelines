"""
Snowflake Feature Store Setup Script

This script initializes and configures a Snowflake Feature Store by:
  - Creating warehouses for feature store operations (default + additional)
  - Initializing the Feature Store in the configured database/schema
  - Registering entities (business objects with join keys)
  - Registering feature views with automated versioning and optional warehouse assignment

Feature views are defined in feature_views.py and referenced by name in config.yml.
Each feature view is automatically versioned based on its definition hash.
Warehouses can be configured in config.yml with one marked as default, and others
can be assigned to specific feature views via the warehouse argument.

Usage:
    python setup_feature_store.py

Configuration:
    Requires feature_store/config.yml with entities, warehouses, and feature_views definitions.
    Requires Snowflake connection configuration via ml_utils.snowflake_env module.
"""

import re
import yaml
import feature_views as feature_views
import os
from typing import Dict, Any, List, Optional
from snowflake.snowpark import Session
from snowflake.ml.feature_store import (
    FeatureStore, 
    CreationMode, 
    Entity, 
    FeatureView,
)
import snowflake.snowpark.functions as F
from ml_utils.snowflake_env import (
    ENVIRONMENT,
    ROLE_NAME,
    DB_NAME,
    SCHEMA_NAME,
    get_session,
)

# Valid warehouse parameter names based on Snowflake SQL reference
VALID_WAREHOUSE_KEYS = [
    "WAREHOUSE_SIZE", "WAREHOUSE_TYPE", "RESOURCE_CONSTRAINT", "MAX_CLUSTER_COUNT", 
    "MIN_CLUSTER_COUNT", "SCALING_POLICY", "AUTO_SUSPEND", "AUTO_RESUME", 
    "INITIALLY_SUSPENDED", "RESOURCE_MONITOR", "COMMENT", "ENABLE_QUERY_ACCELERATION", 
    "QUERY_ACCELERATION_MAX_SCALE_FACTOR", "MAX_CONCURRENCY_LEVEL", 
    "STATEMENT_QUEUED_TIMEOUT_IN_SECONDS", "STATEMENT_TIMEOUT_IN_SECONDS"
]

# Roles to grant privileges to
PRIVILEGE_ROLES = ["ML_ENGINEER", "EXTERNAL_SNOWFLAKE_ARCHITECTS"]

# Privilege definitions based on environment
# DEV: Full access for development and testing
# STAGING/PROD: Read-only and monitoring access
PRIVILEGES_BY_ENV = {
    "DEV": {
        "warehouse": ["ALL PRIVILEGES"],
        "schema": ["USAGE", "CREATE TABLE", "CREATE VIEW", "CREATE DYNAMIC TABLE", "MODIFY"],
        "feature_view_dynamic_table": ["ALL PRIVILEGES"],
        "feature_view_view": ["ALL PRIVILEGES"],  # Allows modifying view query
    },
    "STAGING": {
        "warehouse": ["MONITOR"],
        "schema": ["USAGE"],
        "feature_view_dynamic_table": ["SELECT"],
        "feature_view_view": ["SELECT"],
    },
    "PROD": {
        "warehouse": ["MONITOR"],
        "schema": ["USAGE"],
        "feature_view_dynamic_table": ["SELECT"],
        "feature_view_view": ["SELECT"],
    },
}


def _grant_privileges(session: Session, object_type: str, object_name: str) -> None:
    """
    Grant privileges on a Snowflake object to configured roles based on environment.
    
    Applies environment-appropriate privileges to the ML_ENGINEER and 
    EXTERNAL_SNOWFLAKE_ARCHITECTS roles. DEV environment gets full access,
    while STAGING and PROD environments get monitoring/viewing privileges only.
    
    For feature views, attempts to grant on both DYNAMIC TABLE and VIEW since
    feature views can be materialized as either type, with different privileges
    for each (views get ALL PRIVILEGES in DEV to allow query modification).
    
    Args:
        session (Session): Active Snowflake session
        object_type (str): Type of object (e.g., 'warehouse', 'schema', 'feature_view')
        object_name (str): Fully qualified name of the object
    
    Side Effects:
        Executes GRANT statements in Snowflake
    """
    env_privileges = PRIVILEGES_BY_ENV.get(ENVIRONMENT, PRIVILEGES_BY_ENV["PROD"])
    
    # Feature views can be either dynamic tables or views, try both with their respective privileges
    if object_type == "feature_view":
        object_type_mapping = [
            ("DYNAMIC TABLE", env_privileges.get("feature_view_dynamic_table", [])),
            ("VIEW", env_privileges.get("feature_view_view", [])),
        ]
    else:
        privileges = env_privileges.get(object_type, [])
        if not privileges:
            return
        object_type_mapping = [(object_type.upper(), privileges)]
    
    for sql_object_type, privileges in object_type_mapping:
        for role in PRIVILEGE_ROLES:
            for privilege in privileges:
                try:
                    session.sql(f"""
                        GRANT {privilege} ON {sql_object_type} {object_name} TO ROLE {role};
                    """).collect()
                except Exception as e:
                    # Silently continue if object doesn't exist as this type
                    if "does not exist" not in str(e).lower():
                        print(f"Warning: Could not grant {privilege} on {sql_object_type} {object_name} to {role}: {e}")


def _validate_warehouses(warehouse_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Validate and normalize warehouse configurations from config file.
    
    Processes warehouse configurations to ensure required fields exist,
    validates parameter names against Snowflake's allowed warehouse parameters,
    and identifies the default warehouse.
    
    Args:
        warehouse_configs (List[Dict[str, Any]]): List of warehouse configurations from config.
            Each config should have:
                - name (str): Warehouse name (will be suffixed with environment)
                - default (bool, optional): If True, this is the default feature store warehouse
                - Other optional Snowflake warehouse parameters (e.g., warehouse_size, auto_suspend)
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping warehouse names to their validated parameters.
            The special key "_default" contains the name of the default warehouse.
    
    Raises:
        ValueError: If no default warehouse is specified or if multiple defaults are found.
    
    Example:
        >>> configs = [
        ...     {"name": "MAIN", "warehouse_size": "SMALL", "default": True},
        ...     {"name": "HEAVY", "warehouse_size": "LARGE"}
        ... ]
        >>> _validate_warehouses(configs)
        {
            "_default": "MAIN_DEV_WH",
            "MAIN_DEV_WH": {"WAREHOUSE_SIZE": "SMALL"},
            "HEAVY_DEV_WH": {"WAREHOUSE_SIZE": "LARGE"}
        }
    """
    validated = {}
    default_warehouse = None
    
    for wh_config in warehouse_configs:
        # Sanitize warehouse name and add environment suffix
        base_name = re.sub(r"[^A-Z0-9]", "_", wh_config["name"].upper())
        wh_name = f"{base_name}_{ENVIRONMENT}_WH"
        
        # Check if this is the default warehouse
        if wh_config.get("default", False):
            if default_warehouse is not None:
                raise ValueError(f"Multiple default warehouses specified: {default_warehouse} and {wh_name}")
            default_warehouse = wh_name
        
        # Extract and validate warehouse parameters
        wh_params = {}
        for k, v in wh_config.items():
            if k.upper() in VALID_WAREHOUSE_KEYS:
                wh_params[k.upper()] = v
        
        validated[wh_name] = wh_params
    
    if default_warehouse is None:
        raise ValueError("No default warehouse specified. Set 'default: true' on one warehouse.")
    
    validated["_default"] = default_warehouse
    return validated


def _create_warehouses(session: Session, warehouse_configs: Dict[str, Dict[str, Any]]) -> str:
    """
    Create all configured warehouses in Snowflake.
    
    Creates warehouses using CREATE OR REPLACE to ensure the warehouse exists
    with the specified configuration. Returns the default warehouse name for
    use in Feature Store initialization. Grants appropriate privileges based
    on the current environment.
    
    Args:
        session (Session): Active Snowflake session
        warehouse_configs (Dict[str, Dict[str, Any]]): Validated warehouse configurations
            from _validate_warehouses(), including "_default" key.
    
    Returns:
        str: The name of the default warehouse.
    
    Side Effects:
        Creates or replaces warehouses in Snowflake.
        Grants privileges to ML_ENGINEER and EXTERNAL_SNOWFLAKE_ARCHITECTS roles.
    """
    default_warehouse = warehouse_configs.pop("_default")
    
    for wh_name, wh_params in warehouse_configs.items():
        # Build SQL parameter string
        params_sql = " ".join([f"{k} = {v}" for k, v in wh_params.items()])
        
        # Create warehouse if it doesn't exist (preserves existing warehouse and its usage history)
        session.sql(f"""
            CREATE WAREHOUSE IF NOT EXISTS {wh_name};
        """).collect()
        
        # Update warehouse properties if any are specified
        if params_sql:
            session.sql(f"""
                ALTER WAREHOUSE {wh_name} SET {params_sql};
            """).collect()
        
        # Grant privileges based on environment
        _grant_privileges(session, "warehouse", wh_name)
    
    return default_warehouse


def _get_warehouse_mapping(warehouse_configs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Create a mapping from base warehouse names to full environment-suffixed names.
    
    This allows feature views to reference warehouses by their base name
    (e.g., "HEAVY") while the system uses the full name (e.g., "HEAVY_DEV_WH").
    
    Args:
        warehouse_configs (Dict[str, Dict[str, Any]]): Validated warehouse configurations.
    
    Returns:
        Dict[str, str]: Mapping of base names to full warehouse names.
    """
    mapping = {}
    for wh_name in warehouse_configs.keys():
        if wh_name != "_default":
            # Extract base name by removing the environment suffix
            base_name = wh_name.rsplit(f"_{ENVIRONMENT}_WH", 1)[0]
            mapping[base_name] = wh_name
    return mapping


def _validate_entity(entity_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize entity configuration from config file.
    
    Ensures that the join_keys field is always a list, converting single string
    values to a list with one element for consistency.
    
    Args:
        entity_config (Dict[str, Any]): Entity configuration with fields:
            - name (str): Entity name
            - join_keys (str or List[str]): Column(s) to use as join keys
    
    Returns:
        Dict[str, Any]: Normalized entity configuration with join_keys as a list

    """
    # Ensure join_keys is always a list for consistent handling
    keys = entity_config["join_keys"]
    entity_config["join_keys"] = keys if isinstance(keys, list) else [keys]

    return entity_config

def _validate_featureview(
    feature_view_config: Dict[str, Any], 
    warehouse_mapping: Dict[str, str],
    default_warehouse: str
) -> Dict[str, Any]:
    """
    Validate and normalize feature view configuration from config file.
    
    Extracts required and optional fields, normalizes entities to a list,
    resolves warehouse references to full names, and removes fields with 
    None values to use FeatureView defaults.
    
    Args:
        feature_view_config (Dict[str, Any]): Feature view configuration with fields:
            - name (str): Feature view name
            - function (str): Name of function in feature_views.py
            - entities (str or List[str]): Entity name(s) to join on
            - timestamp_col (str, optional): Timestamp column for point-in-time joins
            - refresh_freq (str, optional): Refresh frequency (e.g., "1 hour")
            - desc (str, optional): Feature view description
            - warehouse (str, optional): Warehouse name for this feature view
        warehouse_mapping (Dict[str, str]): Mapping from base warehouse names to 
            full environment-suffixed names.
        default_warehouse (str): The default warehouse name to use if not specified.
    
    Returns:
        Dict[str, Any]: Normalized configuration with only non-None values, suitable for
                       FeatureView(**config) unpacking

    Raises:
        ValueError: If a specified warehouse is not found in the configuration.
    """
    # Ensure entities is always a list for consistent handling
    entities = feature_view_config["entities"]
    
    # Resolve warehouse reference to full name
    warehouse_ref = feature_view_config.get("warehouse", None)
    if warehouse_ref:
        warehouse_ref_upper = warehouse_ref.upper()
        if warehouse_ref_upper in warehouse_mapping:
            warehouse = warehouse_mapping[warehouse_ref_upper]
        else:
            raise ValueError(
                f"Warehouse '{warehouse_ref}' not found in config. "
                f"Available warehouses: {list(warehouse_mapping.keys())}"
            )
    else:
        warehouse = None
    
    valid_dict = dict(
        name = feature_view_config["name"],
        entities = entities if isinstance(entities, list) else [entities],
        timestamp_col = feature_view_config.get("timestamp_col", None),
        refresh_freq = feature_view_config.get("refresh_freq", None),
        desc = feature_view_config.get("desc", None),
        warehouse = warehouse,
    )
    
    # Remove None values to allow FeatureView to use its defaults
    valid_dict = {k:v for k,v in valid_dict.items() if v}
    return valid_dict

def _version_featureview(feature_store: FeatureStore, feature_view: FeatureView) -> str:
    """
    Determine the appropriate version for a feature view based on its definition.
    
    Implements a smart versioning strategy that:
      - Creates a new version (increments) when breaking changes are detected
        (changes to entities, query, timestamp column, or clustering)
      - Updates the existing version in-place for non-breaking metadata changes
        (refresh frequency, warehouse, or description)
      - Returns version "1" for brand new feature views
    
    This approach minimizes unnecessary version proliferation while ensuring
    that downstream consumers are protected from breaking schema changes.
    
    Args:
        feature_store (FeatureStore): The initialized Snowflake Feature Store instance
        feature_view (FeatureView): The new feature view to version
    
    Returns:
        str: The version string to use when registering the feature view.
             Either a new incremented version or the existing version number.
    
    Examples:
        >>> version = _version_featureview(fs, my_feature_view)
        >>> fs.register_feature_view(my_feature_view, version=version)
    
    """
    name = str(feature_view.name)
    
    # Check if any versions of this feature view already exist
    existing = feature_store.list_feature_views().filter(F.col("NAME") == name).collect()
    
    if existing:
        # Find the highest (most recent) version number
        last_version = max([int(row.VERSION) for row in existing])
        last_feature_view = feature_store.get_feature_view(name=name, version=str(last_version))
        
        # Compare entities - a change in entities is a breaking change
        last_ent = [e.name for e in last_feature_view.entities]
        new_ent = [e.name for e in feature_view.entities]
        if last_ent != new_ent:
            return str(last_version+1)
        
        # Check for breaking changes in core feature view attributes
        # These attributes affect the data schema or query logic
        breaking_change_keys = ['_query', '_name','_timestamp_col','_cluster_by']
        for k in breaking_change_keys:
            if getattr(last_feature_view, k) != getattr(feature_view, k):
                return str(last_version+1)

        # For non-breaking metadata changes, update the existing version in-place
        # These changes don't affect the data schema, so no new version is needed
        metadata_keys = ["refresh_freq", "warehouse", "desc"]
        updates = {
            k: getattr(feature_view, k) 
            for k in metadata_keys
            if getattr(feature_view, k) != getattr(last_feature_view, k)
        }
        if updates:
            feature_store.update_feature_view(name=name, version=str(last_version), **updates)
        
        return str(last_version)
    
    # No existing versions found - this is a new feature view
    return str(1)

if __name__ == "__main__":

    # Load feature store configuration from YAML file
    config = yaml.safe_load(open("feature_store/config.yml","r"))

    # Validate warehouse configurations
    warehouse_configs = _validate_warehouses(config.get("warehouses", []))
    warehouse_mapping = _get_warehouse_mapping(warehouse_configs)

    # Initialize Snowflake session with configured connection
    # Use connection name if provided for local development, otherwise use user, password, and account
    session = get_session()
    session.use_role(ROLE_NAME)
    session.use_database(DB_NAME)
    session.use_schema(SCHEMA_NAME)

    # Create all configured warehouses and get the default warehouse name
    default_warehouse = _create_warehouses(session, warehouse_configs)
    session.use_warehouse(default_warehouse)

    # Initialize Feature Store (creates necessary metadata tables)
    fs = FeatureStore(
        session,
        database=session.get_current_database(),
        name=session.get_current_schema(),
        default_warehouse=default_warehouse,
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )

    # Grant schema-level privileges based on environment
    schema_name = f"{session.get_current_database()}.{session.get_current_schema()}"
    _grant_privileges(session, "schema", schema_name)

    # Register all entities from configuration
    # Entities represent business objects (e.g., users, products) with join keys
    for e in config["entities"]:
        ent_args = _validate_entity(e)
        entity = Entity(**ent_args)
        fs.register_entity(entity)

    # Register all feature views from configuration
    # Each feature view generates features by executing a function from feature_views.py
    for f in config["feature_views"]:
        fv_args = _validate_featureview(f, warehouse_mapping, default_warehouse)
        
        # Dynamically get the feature generation function by name
        feature_func = getattr(feature_views, f["function"])
        
        # Retrieve entity objects and remove from args (will pass separately)
        entities = [fs.get_entity(e) for e in fv_args.pop("entities")]
        
        # Create feature view with generated DataFrame
        fv = FeatureView(
            feature_df=feature_func(session),
            entities=entities,
            **fv_args,
        )
        # Generate version hash based on feature view definition
        version = _version_featureview(fs,fv)
        fv_reg = fs.register_feature_view(fv, version=version)
        
        # Grant privileges on the feature view  based on environment
        fv_full_name = f"{session.get_current_database()}.{session.get_current_schema()}.{fv_args['name']}${version}"
        _grant_privileges(session, "feature_view", fv_full_name)

        # Suspend feature views in non-PROD environments to prevent accidental execution
        if (ENVIRONMENT != "PROD") & (fv.warehouse is not None):
            fs.suspend_feature_view(fv_reg)
    
    session.close()
    print("Feature store setup complete!")


