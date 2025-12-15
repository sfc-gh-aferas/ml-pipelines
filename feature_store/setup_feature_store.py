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
    Requires Snowflake connection configuration via january_ml.constants module.
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
from january_ml.snowflake_env import (
    ENVIRONMENT,
    get_session,
    get_feature_schema,
)

# Valid warehouse parameter names based on Snowflake SQL reference
VALID_WAREHOUSE_KEYS = [
    "WAREHOUSE_SIZE", "WAREHOUSE_TYPE", "RESOURCE_CONSTRAINT", "MAX_CLUSTER_COUNT", 
    "MIN_CLUSTER_COUNT", "SCALING_POLICY", "AUTO_SUSPEND", "AUTO_RESUME", 
    "INITIALLY_SUSPENDED", "RESOURCE_MONITOR", "COMMENT", "ENABLE_QUERY_ACCELERATION", 
    "QUERY_ACCELERATION_MAX_SCALE_FACTOR", "MAX_CONCURRENCY_LEVEL", 
    "STATEMENT_QUEUED_TIMEOUT_IN_SECONDS", "STATEMENT_TIMEOUT_IN_SECONDS"
]


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
    use in Feature Store initialization.
    
    Args:
        session (Session): Active Snowflake session
        warehouse_configs (Dict[str, Dict[str, Any]]): Validated warehouse configurations
            from _validate_warehouses(), including "_default" key.
    
    Returns:
        str: The name of the default warehouse.
    
    Side Effects:
        Creates or replaces warehouses in Snowflake.
    """
    default_warehouse = warehouse_configs.pop("_default")
    
    for wh_name, wh_params in warehouse_configs.items():
        # Build SQL parameter string
        params_sql = " ".join([f"{k} = {v}" for k, v in wh_params.items()])
        
        session.sql(f"""
            CREATE OR REPLACE WAREHOUSE {wh_name} {params_sql};
        """).collect()
        print(f"Created warehouse: {wh_name}")
    
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
        warehouse = default_warehouse
    
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
    session.use_schema(get_feature_schema(session))

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
        fs.register_feature_view(fv, version=version)
    
    session.close()
    print("Feature store setup complete!")


