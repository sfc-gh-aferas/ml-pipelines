"""
Snowflake ML Project Cleanup Script

This script removes all resources created for a given project:
  - Compute pool (PROJECT_ENVIRONMENT_COMPUTE)
  - Warehouse (PROJECT_ENVIRONMENT_WH)
  - Notebooks (project__notebook_name)
  - DAGs/Tasks (project_dag_name)
  - Staged files in BUILD_STAGE and JOB_STAGE

It can also clean up feature store resources:
  - Feature views (with all versions)
  - Entities (if no longer referenced)
  - Feature store warehouse

Usage:
    python scripts/cleanup.py <project_name> [--dry-run]
    python scripts/cleanup.py --features <feature1> <feature2> [--dry-run]
    python scripts/cleanup.py --all [--dry-run]

Environment:
    Requires Snowflake connection configuration via january_ml.snowflake_env module.
"""
import os
import re
import yaml
import argparse
from snowflake.snowpark.session import Session
import snowflake.snowpark.functions as F
from january_ml.snowflake_env import (
    ENVIRONMENT,
    get_session,
    get_model_schema,
    get_feature_schema,
    get_build_stage,
    get_job_stage,
)


FEATURE_CONFIG_PATH = "feature_store/config.yml"
FEATURE_STORE_WAREHOUSE = f"FEATURE_STORE_{ENVIRONMENT}_WH"


def load_feature_config() -> dict:
    """Load feature store configuration from YAML file."""
    if not os.path.exists(FEATURE_CONFIG_PATH):
        return {"entities": [], "feature_views": []}
    return yaml.safe_load(open(FEATURE_CONFIG_PATH, "r"))


def get_all_project_names() -> list[str]:
    """Get all project names from the projects directory."""
    projects_dir = "projects"
    if not os.path.exists(projects_dir):
        return []
    return [
        d for d in os.listdir(projects_dir)
        if os.path.isdir(os.path.join(projects_dir, d)) and not d.startswith("__")
    ]


def cleanup_staged_files(session: Session, project_name: str, dry_run: bool = False) -> None:
    """Remove project files from BUILD_STAGE and JOB_STAGE."""
    BUILD_STAGE = get_build_stage(session)
    JOB_STAGE = get_job_stage(session)
    
    for stage in [BUILD_STAGE, JOB_STAGE]:
        sql = f"REMOVE @{stage}/{project_name}"
        if dry_run:
            print(f"  [DRY RUN] Would execute: {sql}")
        else:
            try:
                session.sql(sql).collect()
                print(f"  ✅ Removed files from {stage}/{project_name}")
            except Exception as e:
                print(f"  ⚠️  Could not remove files from {stage}/{project_name}: {e}")


def cleanup_notebooks(session: Session, project_name: str, dry_run: bool = False) -> None:
    """Remove all notebooks associated with the project."""
    db = session.get_current_database()
    schema = session.get_current_schema()
    
    # Find all notebooks matching the project pattern
    notebooks = session.sql(f"""
        SHOW NOTEBOOKS LIKE '{project_name}__%' IN SCHEMA {db}.{schema}
    """).collect()
    
    if not notebooks:
        print("  ℹ️  No notebooks found for this project")
        return
    
    for nb in notebooks:
        notebook_name = nb["name"]
        sql = f"DROP NOTEBOOK IF EXISTS {db}.{schema}.{notebook_name}"
        if dry_run:
            print(f"  [DRY RUN] Would execute: {sql}")
        else:
            try:
                session.sql(sql).collect()
                print(f"  ✅ Dropped notebook {notebook_name}")
            except Exception as e:
                print(f"  ⚠️  Could not drop notebook {notebook_name}: {e}")


def cleanup_dags(session: Session, project_name: str, dag_names: list[str], dry_run: bool = False) -> None:
    """Remove all DAGs/tasks associated with the project."""
    db = session.get_current_database()
    schema = session.get_current_schema()
    
    for dag_name in dag_names:
        full_dag_name = f"{project_name}_{dag_name}"
        
        # First suspend the task (root task of the DAG)
        suspend_sql = f"ALTER TASK IF EXISTS {db}.{schema}.{full_dag_name} SUSPEND"
        drop_sql = f"DROP TASK IF EXISTS {db}.{schema}.{full_dag_name}"
        
        if dry_run:
            print(f"  [DRY RUN] Would execute: {suspend_sql}")
            print(f"  [DRY RUN] Would execute: {drop_sql}")
        else:
            try:
                session.sql(suspend_sql).collect()
                session.sql(drop_sql).collect()
                print(f"  ✅ Dropped DAG/task {full_dag_name}")
            except Exception as e:
                print(f"  ⚠️  Could not drop DAG/task {full_dag_name}: {e}")


def cleanup_compute_pool(session: Session, project_name: str, dry_run: bool = False) -> None:
    """Remove the project's compute pool."""
    sanitized_name = re.sub(r"[^A-Z0-9]", "_", project_name.upper())
    compute_pool_name = f"{sanitized_name}_{ENVIRONMENT}_COMPUTE"
    
    # First stop all services and drop the pool
    stop_sql = f"ALTER COMPUTE POOL IF EXISTS {compute_pool_name} STOP ALL"
    drop_sql = f"DROP COMPUTE POOL IF EXISTS {compute_pool_name}"
    
    if dry_run:
        print(f"  [DRY RUN] Would execute: {stop_sql}")
        print(f"  [DRY RUN] Would execute: {drop_sql}")
    else:
        try:
            session.sql(stop_sql).collect()
            session.sql(drop_sql).collect()
            print(f"  ✅ Dropped compute pool {compute_pool_name}")
        except Exception as e:
            print(f"  ⚠️  Could not drop compute pool {compute_pool_name}: {e}")


def cleanup_warehouse(session: Session, project_name: str, dry_run: bool = False) -> None:
    """Remove the project's warehouse."""
    sanitized_name = re.sub(r"[^A-Z0-9]", "_", project_name.upper())
    warehouse_name = f"{sanitized_name}_{ENVIRONMENT}_WH"
    
    sql = f"DROP WAREHOUSE IF EXISTS {warehouse_name}"
    
    if dry_run:
        print(f"  [DRY RUN] Would execute: {sql}")
    else:
        try:
            session.sql(sql).collect()
            print(f"  ✅ Dropped warehouse {warehouse_name}")
        except Exception as e:
            print(f"  ⚠️  Could not drop warehouse {warehouse_name}: {e}")


def cleanup_feature_view(session: Session, feature_name: str, dry_run: bool = False) -> None:
    """Remove a feature view and all its versions from the feature store."""
    from snowflake.ml.feature_store import FeatureStore, CreationMode
    
    db = session.get_current_database()
    schema = session.get_current_schema()
    
    # Initialize feature store in read mode
    try:
        fs = FeatureStore(
            session,
            database=db,
            name=schema,
            default_warehouse=FEATURE_STORE_WAREHOUSE,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
    except Exception as e:
        print(f"  ⚠️  Could not initialize feature store: {e}")
        return
    
    # Find all versions of the feature view
    try:
        existing = fs.list_feature_views().filter(F.col("NAME") == feature_name.upper()).collect()
    except Exception as e:
        print(f"  ⚠️  Could not list feature views: {e}")
        return
    
    if not existing:
        print(f"  ℹ️  No feature view found with name: {feature_name}")
        return
    
    # Delete each version of the feature view
    for row in existing:
        version = row.VERSION
        if dry_run:
            print(f"  [DRY RUN] Would delete feature view {feature_name} version {version}")
        else:
            try:
                fs.delete_feature_view(fs.get_feature_view(name=feature_name, version=str(version)))
                print(f"  ✅ Deleted feature view {feature_name} version {version}")
            except Exception as e:
                print(f"  ⚠️  Could not delete feature view {feature_name} version {version}: {e}")


def cleanup_entity(session: Session, entity_name: str, dry_run: bool = False) -> None:
    """Remove an entity from the feature store if it has no dependent feature views."""
    from snowflake.ml.feature_store import FeatureStore, CreationMode
    
    db = session.get_current_database()
    schema = session.get_current_schema()
    
    try:
        fs = FeatureStore(
            session,
            database=db,
            name=schema,
            default_warehouse=FEATURE_STORE_WAREHOUSE,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
    except Exception as e:
        print(f"  ⚠️  Could not initialize feature store: {e}")
        return
    
    if dry_run:
        print(f"  [DRY RUN] Would delete entity {entity_name}")
    else:
        try:
            entity = fs.get_entity(entity_name)
            fs.delete_entity(entity_name)
            print(f"  ✅ Deleted entity {entity_name}")
        except Exception as e:
            # Entity might not exist or might have dependencies
            print(f"  ⚠️  Could not delete entity {entity_name}: {e}")


def cleanup_feature_store_warehouse(session: Session, dry_run: bool = False) -> None:
    """Remove the feature store warehouse."""
    sql = f"DROP WAREHOUSE IF EXISTS {FEATURE_STORE_WAREHOUSE}"
    
    if dry_run:
        print(f"  [DRY RUN] Would execute: {sql}")
    else:
        try:
            session.sql(sql).collect()
            print(f"  ✅ Dropped warehouse {FEATURE_STORE_WAREHOUSE}")
        except Exception as e:
            print(f"  ⚠️  Could not drop warehouse {FEATURE_STORE_WAREHOUSE}: {e}")


def cleanup_project(session: Session, project_name: str, dry_run: bool = False) -> None:
    """Clean up all resources for a single project."""
    # Load project configuration to get DAG names
    config_path = f"projects/{project_name}/config.yml"
    if not os.path.exists(config_path):
        print(f"  ⚠️  Project config not found: {config_path}")
        print("     Continuing with cleanup of resources that can be discovered...")
        dag_names = []
    else:
        config = yaml.safe_load(open(config_path, "r"))
        dag_names = [d["name"] for d in config.get("deploy", {}).get("DAGS", [])]
    
    # Cleanup in reverse order of creation
    print("  📋 Removing DAGs/Tasks...")
    if dag_names:
        cleanup_dags(session, project_name, dag_names, dry_run)
    else:
        print("    ℹ️  No DAG names found in config, skipping...")

    print("  📓 Removing Notebooks...")
    cleanup_notebooks(session, project_name, dry_run)

    print("  📦 Removing Staged Files...")
    cleanup_staged_files(session, project_name, dry_run)

    print("  🖥️  Removing Compute Pool...")
    cleanup_compute_pool(session, project_name, dry_run)

    print("  🏭 Removing Warehouse...")
    cleanup_warehouse(session, project_name, dry_run)


def cleanup_features(session: Session, feature_names: list[str], dry_run: bool = False) -> None:
    """Clean up specified feature views and their associated entities."""
    feature_config = load_feature_config()
    
    # Build lookup for feature views and their entities
    fv_to_entities = {}
    for fv in feature_config.get("feature_views", []):
        entities = fv.get("entities", [])
        if isinstance(entities, str):
            entities = [entities]
        fv_to_entities[fv["name"]] = entities
    
    # Validate feature names
    valid_features = set(fv_to_entities.keys())
    for name in feature_names:
        if name not in valid_features:
            print(f"  ⚠️  Feature '{name}' not found in {FEATURE_CONFIG_PATH}")
            print(f"     Available features: {', '.join(valid_features) or 'none'}")
    
    # Delete feature views
    print("  📊 Removing Feature Views...")
    for name in feature_names:
        if name in valid_features:
            cleanup_feature_view(session, name, dry_run)
    
    # Collect entities that might need cleanup
    entities_to_check = set()
    for name in feature_names:
        if name in fv_to_entities:
            entities_to_check.update(fv_to_entities[name])
    
    # Find entities that are still in use by other feature views
    remaining_fv_names = valid_features - set(feature_names)
    entities_still_in_use = set()
    for fv_name in remaining_fv_names:
        entities_still_in_use.update(fv_to_entities.get(fv_name, []))
    
    # Only delete entities that are no longer in use
    entities_to_delete = entities_to_check - entities_still_in_use
    
    if entities_to_delete:
        print("  🏷️  Removing Entities (no longer in use)...")
        for entity_name in entities_to_delete:
            cleanup_entity(session, entity_name, dry_run)


def main():
    parser = argparse.ArgumentParser(
        prog="cleanup.py",
        description="Remove Snowflake resources created for projects and/or feature store.",
    )

    parser.add_argument(
        "project_name",
        nargs="?",
        default=None,
        help="The name of the project to clean up",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        metavar="FEATURE",
        help="Feature view names to clean up (from feature_store/config.yml)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        dest="cleanup_all",
        help="Clean up all projects and all features",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be deleted without actually deleting anything",
    )

    args = parser.parse_args()
    dry_run = args.dry_run

    # Validate arguments
    if not args.project_name and not args.features and not args.cleanup_all:
        parser.error("Must specify a project name, --features, or --all")

    if args.cleanup_all and (args.project_name or args.features):
        parser.error("--all cannot be combined with project_name or --features")

    # Initialize Snowflake session
    session = get_session()

    print(f"\n{'=' * 60}")
    print(f"🧹 Snowflake ML Cleanup")
    print(f"   Environment: {ENVIRONMENT}")
    print(f"   Database: {session.get_current_database()}")
    if dry_run:
        print(f"   Mode: DRY RUN (no changes will be made)")
    print(f"{'=' * 60}\n")

    # Handle --all flag
    if args.cleanup_all:
        # Clean up all projects
        project_names = get_all_project_names()
        if project_names:
            print("📁 Cleaning up all projects...")
            session.use_schema(get_model_schema(session))
            for project_name in project_names:
                print(f"\n🧹 Project: {project_name}")
                cleanup_project(session, project_name, dry_run)
        else:
            print("ℹ️  No projects found in projects/ directory")

        # Clean up all features
        feature_config = load_feature_config()
        feature_names = [fv["name"] for fv in feature_config.get("feature_views", [])]
        
        if feature_names:
            print(f"\n📊 Cleaning up all features...")
            session.use_schema(get_feature_schema(session))
            cleanup_features(session, feature_names, dry_run)
            
            # Clean up all entities
            entity_names = [e["name"] for e in feature_config.get("entities", [])]
            if entity_names:
                print("\n🏷️  Removing all Entities...")
                for entity_name in entity_names:
                    cleanup_entity(session, entity_name, dry_run)
            
            # Clean up feature store warehouse
            print("\n🏭 Removing Feature Store Warehouse...")
            cleanup_feature_store_warehouse(session, dry_run)
        else:
            print("ℹ️  No features found in feature_store/config.yml")

    # Handle --features flag
    elif args.features:
        print(f"📊 Cleaning up features: {', '.join(args.features)}")
        session.use_schema(get_feature_schema(session))
        cleanup_features(session, args.features, dry_run)

    # Handle project cleanup
    elif args.project_name:
        print(f"🧹 Cleaning up project: {args.project_name}")
        session.use_schema(get_model_schema(session))
        cleanup_project(session, args.project_name, dry_run)

    print(f"\n{'=' * 60}")
    if dry_run:
        print("✅ Dry run complete. No resources were modified.")
    else:
        print("✅ Cleanup complete!")
    print(f"{'=' * 60}\n")

    session.close()


if __name__ == "__main__":
    main()
