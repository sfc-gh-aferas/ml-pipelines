"""
Snowflake ML Project Cleanup Script

This script removes all resources created for a given project:
  - Compute pool (PROJECT_ENVIRONMENT_COMPUTE)
  - Warehouse (PROJECT_ENVIRONMENT_WH)
  - Notebooks (project__notebook_name)
  - DAGs/Tasks (project_dag_name)
  - Staged files in BUILD_STAGE and JOB_STAGE

Usage:
    python scripts/cleanup.py <project_name> [--dry-run]

Environment:
    Requires Snowflake connection configuration via january_ml.snowflake_env module.
"""
import os
import re
import yaml
import argparse
from snowflake.snowpark.session import Session
from january_ml.snowflake_env import (
    ENVIRONMENT,
    get_session,
    get_model_schema,
    get_build_stage,
    get_job_stage,
)


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


def main():
    parser = argparse.ArgumentParser(
        prog="cleanup.py",
        description="Remove all Snowflake resources created for a project.",
    )

    parser.add_argument(
        "project_name",
        help="The name of the project to clean up",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be deleted without actually deleting anything",
    )

    args = parser.parse_args()
    project_name = args.project_name
    dry_run = args.dry_run

    # Load project configuration to get DAG names
    config_path = f"projects/{project_name}/config.yml"
    if not os.path.exists(config_path):
        print(f"❌ Project config not found: {config_path}")
        print("   Continuing with cleanup of resources that can be discovered...")
        dag_names = []
    else:
        config = yaml.safe_load(open(config_path, "r"))
        dag_names = [d["name"] for d in config.get("deploy", {}).get("DAGS", [])]

    # Initialize Snowflake session
    session = get_session()
    session.use_schema(get_model_schema(session))

    print(f"\n{'=' * 60}")
    print(f"🧹 Cleaning up project: {project_name}")
    print(f"   Environment: {ENVIRONMENT}")
    print(f"   Database: {session.get_current_database()}")
    print(f"   Schema: {session.get_current_schema()}")
    if dry_run:
        print(f"   Mode: DRY RUN (no changes will be made)")
    print(f"{'=' * 60}\n")

    # Cleanup in reverse order of creation
    print("📋 Removing DAGs/Tasks...")
    if dag_names:
        cleanup_dags(session, project_name, dag_names, dry_run)
    else:
        print("  ℹ️  No DAG names found in config, skipping...")

    print("\n📓 Removing Notebooks...")
    cleanup_notebooks(session, project_name, dry_run)

    print("\n📦 Removing Staged Files...")
    cleanup_staged_files(session, project_name, dry_run)

    print("\n🖥️  Removing Compute Pool...")
    cleanup_compute_pool(session, project_name, dry_run)

    print("\n🏭 Removing Warehouse...")
    cleanup_warehouse(session, project_name, dry_run)

    print(f"\n{'=' * 60}")
    if dry_run:
        print("✅ Dry run complete. No resources were modified.")
    else:
        print(f"✅ Cleanup complete for project: {project_name}")
    print(f"{'=' * 60}\n")

    session.close()


if __name__ == "__main__":
    main()
