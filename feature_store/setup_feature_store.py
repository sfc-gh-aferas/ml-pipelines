"""
Snowflake Feature Store Setup Script

This script initializes and configures a Snowflake Feature Store by:
  - Creating a dedicated warehouse for feature store operations
  - Initializing the Feature Store in the configured database/schema
  - Registering entities (business objects with join keys)
  - Registering feature views with automated versioning

Feature views are defined in feature_views.py and referenced by name in config.yml.
Each feature view is automatically versioned based on its definition hash.

Usage:
    python setup_feature_store.py

Configuration:
    Requires feature_store/config.yml with entities and feature_views definitions.
    Requires Snowflake connection configuration via january_ml.constants module.
"""

import yaml
import feature_views
from typing import Dict, Any, List, Union
from snowflake.snowpark import Session
from snowflake.ml.feature_store import (
    FeatureStore, 
    CreationMode, 
    Entity, 
    FeatureView,
)

from january_ml.utils import version_featureview
from january_ml.constants import (
    CONNECTION,
    DB_NAME,
    FEATURE_SCHEMA,
    ROLE_NAME,
    ENVIRONMENT,
)

# Create environment-specific warehouse name for feature store operations
WAREHOUSE = f"FEATURE_STORE_{ENVIRONMENT}_WH"

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

def _validate_featureview(feature_view_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize feature view configuration from config file.
    
    Extracts required and optional fields, normalizes entities to a list,
    and removes fields with None values to use FeatureView defaults.
    
    Args:
        feature_view_config (Dict[str, Any]): Feature view configuration with fields:
            - name (str): Feature view name
            - function (str): Name of function in feature_views.py
            - entities (str or List[str]): Entity name(s) to join on
            - timestamp_col (str, optional): Timestamp column for point-in-time joins
            - refresh_freq (str, optional): Refresh frequency (e.g., "1 hour")
            - desc (str, optional): Feature view description
    
    Returns:
        Dict[str, Any]: Normalized configuration with only non-None values, suitable for
                       FeatureView(**config) unpacking

    """
    # Ensure entities is always a list for consistent handling
    entities = feature_view_config["entities"]
    valid_dict = dict(
        name = feature_view_config["name"],
        entities = entities if isinstance(entities, list) else [entities],
        timestamp_col = feature_view_config.get("timestamp_col",None),
        refresh_freq = feature_view_config.get("refresh_freq",None),
        desc = feature_view_config.get("desc",None),
    )
    
    # Remove None values to allow FeatureView to use its defaults
    valid_dict = {k:v for k,v in valid_dict.items() if v}
    return valid_dict

if __name__ == "__main__":

    # Load feature store configuration from YAML file
    config = yaml.safe_load(open("feature_store/config.yml","r"))

    # Initialize Snowflake session with configured connection
    session = Session.builder.config("connection_name",CONNECTION).getOrCreate()
    session.use_role(ROLE_NAME)
    session.use_database(DB_NAME)
    session.use_schema(FEATURE_SCHEMA)

    # Create dedicated warehouse for feature store operations
    session.sql(f"""
        CREATE WAREHOUSE IF NOT EXISTS {WAREHOUSE}
            WAREHOUSE_SIZE = SMALL;
    """).collect()
    session.use_warehouse(WAREHOUSE)

    # Initialize Feature Store (creates necessary metadata tables)
    fs = FeatureStore(
        session,
        database=DB_NAME,
        name=FEATURE_SCHEMA,
        default_warehouse=WAREHOUSE,
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
        fv_args = _validate_featureview(f)
        
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
        version = version_featureview(fv)
        fs.register_feature_view(fv, version=version)
    
    session.close()
    print("Feature store setup complete!")


