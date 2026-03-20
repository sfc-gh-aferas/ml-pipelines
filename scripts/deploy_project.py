"""
Snowflake ML Project Deployment Script

This script deploys machine learning projects to Snowflake by:
  - Staging project files and dependencies to Snowflake stages
  - Creating and deploying notebooks (.ipynb files) with runtime configuration
  - Creating ML jobs from Python scripts (.py files) using Snowflake ML
  - Orchestrating tasks into DAGs (Directed Acyclic Graphs) with scheduling
  - Optionally executing deployed DAGs immediately for validation

Designed for use in CI/CD pipelines (e.g., GitHub Actions) to automate
deployment of ML workflows to Snowflake environments.

Usage:
    python deploy_project.py <project_name> [--run-dag]

Environment:
    Requires Snowflake connection configuration via ml_utils.snowflake_env module.
"""
import os
import sys
import json
import time
import importlib
import re
import yaml
from datetime import timedelta
from typing import Union
from collections.abc import Callable
from typing import Union
from snowflake.snowpark.session import Session
from snowflake.core import CreateMode, Root
from snowflake.ml.jobs import submit_from_stage
from snowflake.core.task.dagv1 import DAG, DAGTask, DAGOperation
from snowflake.core.task.context import TaskContext
from snowflake.core.task import Cron
from ml_utils.snowflake_env import (
    get_session,
    ENVIRONMENT,
    BUILD_STAGE,
    JOB_STAGE,
    ROLE_NAME,
    DB_NAME,
    SCHEMA_NAME
)

# Roles to grant privileges to
PRIVILEGE_ROLES = ["ACCOUNTADMIN"]

# Privilege definitions based on environment
# DEV: Full access for development and testing
# STAGING/PROD: Read-only and monitoring access
PRIVILEGES_BY_ENV = {
    "DEV": {
        "warehouse": ["ALL PRIVILEGES"],
        "compute pool": ["ALL PRIVILEGES"],
        "stage": ["ALL PRIVILEGES"],
        "task": ["ALL PRIVILEGES"],
        "schema": ["USAGE", "CREATE STAGE", "CREATE NOTEBOOK", "CREATE TASK", "MODIFY"],
    },
    "STAGING": {
        "warehouse": ["MONITOR"],
        "compute pool": ["MONITOR"],
        "stage": ["READ"],
        "task": ["MONITOR"],
        "schema": ["USAGE"],
    },
    "PROD": {
        "warehouse": ["MONITOR"],
        "compute pool": ["MONITOR"],
        "stage": ["READ"],
        "task": ["MONITOR"],
        "schema": ["USAGE"],
    },
}


def _grant_privileges(session: Session, object_type: str, object_name: str) -> None:
    """
    Grant privileges on a Snowflake object to configured roles based on environment.
    
    Applies environment-appropriate privileges to the ACCOUNTADMIN role.
    DEV environment gets full access, while STAGING and PROD environments
    get monitoring/viewing privileges only.
    
    Args:
        session (Session): Active Snowflake session
        object_type (str): Type of object (e.g., 'warehouse', 'stage', 'task', etc.)
        object_name (str): Fully qualified name of the object
    
    Side Effects:
        Executes GRANT statements in Snowflake
    """
    env_privileges = PRIVILEGES_BY_ENV.get(ENVIRONMENT, PRIVILEGES_BY_ENV["PROD"])
    privileges = env_privileges.get(object_type, [])
    
    if not privileges:
        return
    
    sql_object_type = object_type.upper()
    
    for role in PRIVILEGE_ROLES:
        for privilege in privileges:
            try:
                session.sql(f"""
                    GRANT {privilege} ON {sql_object_type} {object_name} TO ROLE {role};
                """).collect()
            except Exception as e:
                print(f"Warning: Could not grant {privilege} on {sql_object_type} {object_name} to {role}: {e}")

def _wait_for_run_to_complete(session: Session, dag: DAG) -> str:
    """
    Wait for a DAG run to complete and return the final status.

    This function monitors the most recent DAG run and waits for it to complete.
    It uses exponential backoff to poll the task graph status and returns the final result.

    Args:
        session (Session): Snowflake session object
        dag (DAG): The DAG object to monitor

    Returns:
        str: The final status of the DAG run (e.g., "SUCCEEDED", "FAILED")

    Raises:
        RuntimeError: If no recent runs are found for the DAG
    """
    # NOTE: We assume the most recent run is our run
    # It would be better to add some unique identifier to the DAG to make it easier to identify the run
    DB_NAME = session.get_current_database().replace('"','')
    MODEL_SCHEMA = session.get_current_schema().replace('"','')
    recent_runs = session.sql(
        f"""
        select run_id
            from table({DB_NAME}.information_schema.current_task_graphs(
                root_task_name => '{dag.name.upper()}'
            ))
            where database_name = '{DB_NAME}'
            and schema_name = '{MODEL_SCHEMA}'
            and scheduled_from = 'EXECUTE TASK';
        """,
    ).collect()
    if len(recent_runs) == 0:
        raise RuntimeError("No recent runs found. Did the DAG fail to run?")
    run_id = recent_runs[0][0]
    print(f"DAG runId: {run_id}")

    start_time = time.time()
    dag_result = None
    while dag_result is None:
        result = session.sql(
            f"""
            select state, FIRST_ERROR_MESSAGE
                from table({DB_NAME}.information_schema.complete_task_graphs(
                    root_task_name=>'{dag.name.upper()}'
                ))
                where database_name = '{DB_NAME}'
                and schema_name = '{MODEL_SCHEMA}'
                and run_id = {run_id};
            """,
        ).collect()

        if len(result) > 0:
            dag_result = result[0].STATE
            error_message = result[0].FIRST_ERROR_MESSAGE
            print(error_message)
            print(
                f"DAG completed after {(time.time() - start_time):.2f} seconds with result {dag_result}"
            )
            break

        wait_time = min(
            2 ** ((time.time() - start_time) / 10), 5
        )  # Exponential backoff capped at 5 seconds
        time.sleep(wait_time)

    return dag_result

def stage_directory(session: Session, project_dir: str) -> list[str]:
    """
    Upload project files and dependencies to Snowflake stages.
    
    Removes any previous project files from BUILD_STAGE and JOB_STAGE,
    then uploads all files from the project directory along with the
    ml_utils package wheel. Grants appropriate privileges based on
    the current environment.
    
    Args:
        session (Session): Active Snowflake session
        project_dir (str): Name of the project subdirectory under 'projects/'
    
    Returns:
        list[str]: List of stage paths for all uploaded files (e.g., '@BUILD_STAGE/project/file.py')
    """
    # remove previous project files
    session.sql(f"REMOVE @{BUILD_STAGE}/{project_dir}").collect()
    session.sql(f"REMOVE @{JOB_STAGE}/{project_dir}").collect()
    
    # Grant privileges on stages based on environment
    _grant_privileges(session, "stage", BUILD_STAGE)
    _grant_privileges(session, "stage", JOB_STAGE)

    # Upload all non-directory files from the project directory
    staged_files = []
    for f in os.listdir(f"projects/{project_dir}"):
        filename = f"projects/{project_dir}/{f}"
        if not os.path.isdir(filename):
            result = session.file.put(filename,f"{BUILD_STAGE}/{project_dir}",overwrite=True, auto_compress=False)
            staged_files.append(f"@{BUILD_STAGE}/{project_dir}/{result[0].target}")
    
    # Upload the ml_utils package wheel file
    result = session.file.put(
        "dist/ml_utils-0.0.1-py3-none-any.whl",
        f"{BUILD_STAGE}/{project_dir}/dist", 
        overwrite=True, 
        auto_compress=False
    )
    staged_files.append(f"@{BUILD_STAGE}/{project_dir}/dist/{result[0].target}")
    print(f"{project_dir} uploaded to {BUILD_STAGE}")
    return staged_files

def _deploy_notebook(session: Session, project_name: str) -> str:
    """
    Deploy a notebook project.
    
    Creates a Snowflake Notebook Project from a directory in BUILD_STAGE.
    
    Args:
        session (Session): Active Snowflake session
        project_name (str): Project name used as the notebook object name
    
    Returns:
        str: Fully qualified name of the deployed notebook (DB.SCHEMA.PROJECT_NAME)
    
    """
    # Create fully qualified notebook name with project namespace
    fully_qualified_name = f"{session.get_current_database()}.{session.get_current_schema()}.{project_name}"
    
    # Create notebook with runtime configuration
    nb_sql = f"""
        CREATE OR REPLACE NOTEBOOK PROJECT {fully_qualified_name}
        FROM @{BUILD_STAGE}/{project_name}
    """
    results = session.sql(nb_sql).collect()

    print(f"Successfully deployed notebook {fully_qualified_name}")
    return fully_qualified_name

def _get_return_vals(task_context: TaskContext, return_from_tasks: list, script_args: bool = False) -> Union[list,dict]:
    """
    Retrieve and parse return values from predecessor tasks in a DAG.
    
    Extracts return values from specified predecessor tasks and formats them
    either as command-line arguments (for scripts) or as a dictionary (for functions).
    Handles empty or None results gracefully by skipping them.
    
    Args:
        task_context (TaskContext): Current task execution context
        return_from_tasks (list): List of predecessor task names to get values from
        script_args (bool): If True, format as CLI args (--key value); if False, return as dict
    
    Returns:
        Union[list, dict]: Either a list of CLI arguments ['--key1', 'val1', '--key2', 'val2']
                          or a merged dictionary {'key1': 'val1', 'key2': 'val2'}
    
    Note:
        TODO: Add validation for return value format and task existence
    """
    kw = [] if script_args else {}
    for task in return_from_tasks:
        result = task_context.get_predecessor_return_value(task).replace("'",'"')
        if result:
            try:
                val = json.loads(result)
            except json.JSONDecodeError:
                raise ValueError(f"Return value from task {task} is not a dictionary")
            if not isinstance(val, dict):
                raise ValueError(f"Return value from task {task} is not a dictionary")
            if script_args:
                kw += [i for k,v in val.items() for i in ("--"+str(k),str(v))]
            else:
                kw.update(val)         
    return kw

def _get_notebook_sql(session:Session, fully_qualified_name: str, notebook_file:str, return_from_tasks: list = []) -> Callable:
    """
    Create a SQL string that executes a deployed Snowflake Notebook.
    
    Builds an EXECUTE NOTEBOOK PROJECT statement with parameters from
    predecessor tasks. Handles cases where no parameters are provided.
    
    Args:
        session (Session): Active Snowflake session (used to read predecessor return values)
        fully_qualified_name (str): Full notebook name (DB.SCHEMA.NOTEBOOK_NAME)
        notebook_file (str): Name of the .ipynb file to execute as MAIN_FILE
        return_from_tasks (list): Task names to retrieve parameters from
    
    Returns:
        str: SQL string for EXECUTE NOTEBOOK PROJECT
    """
    ctx = TaskContext(session)
    # Get parameters from predecessor tasks and format as SQL arguments
    params = _get_return_vals(task_context=ctx, return_from_tasks=return_from_tasks, script_args=True)
    params = params + ["--snowflake-env",ENVIRONMENT]
    params = " ".join(params) if params else ""
    nb_exec_sql = f"""
        EXECUTE NOTEBOOK PROJECT {fully_qualified_name}
            MAIN_FILE = '{notebook_file}'
            QUERY_WAREHOUSE = {WAREHOUSE}
            RUNTIME = 'V2.3-CPU-PY3.10'
            COMPUTE_POOL = {COMPUTE_POOL}
            ARGUMENTS = '{params}';
    """        
    return nb_exec_sql

def _get_mljob_runner(filename: str, project_name: str, return_from_tasks: list = []) -> Callable:
    """
    Create a task function that submits and runs a Python script as a Snowflake ML Job.
    
    Returns a callable that submits a Python script from BUILD_STAGE as a Snowflake ML Job
    with dependencies and parameters from predecessor tasks. The job runs on the dynamically
    created COMPUTE_POOL with isolated container execution. Includes pip requirements from
    the project's pip-requirements.txt and the ml_utils package wheel.
    
    Args:
        filename (str): Python script filename to execute (e.g., 'training.py')
        project_name (str): Project name for stage path resolution
        return_from_tasks (list): Task names to retrieve parameters from
    
    Returns:
        Callable: Function that submits and waits for the ML job when called with a session
    """
    def job_func(session: Session) ->  str:
        ctx = TaskContext(session)
        # Get command-line arguments from predecessor tasks
        params = _get_return_vals(task_context=ctx, return_from_tasks=return_from_tasks, script_args=True)
            
        # Submit Python script as ML job with dependencies

        stage_path = f"@{BUILD_STAGE}/{project_name}"
        job = submit_from_stage(
            source=stage_path,
            compute_pool=COMPUTE_POOL,
            entrypoint=f"{filename}",
            stage_name=JOB_STAGE,
            session=session,
            args=params,
            pip_requirements=["-r ../app/pip-requirements.txt", "../app/dist/ml_utils-0.0.1-py3-none-any.whl"],
        )
        # Store and return job results for downstream tasks
        results = job.result() if job.result() else ""
        ctx.set_return_value(results)
        return job.result()

    return job_func

def _get_func_runner(filename: str, return_from_tasks: list = []) -> Callable:
    """
    Create a task function that executes a Python module's main() function.
    
    Returns a callable that imports and runs a Python module's main() function
    with parameters from predecessor tasks. Used for lightweight Python tasks
    that don't require ML Job submission. Adds the ml_utils wheel to sys.path
    and passes the Snowflake session to the module's main function.
    
    Args:
        filename (str): Python module filename (e.g., 'utils.py')
        return_from_tasks (list): Task names to retrieve parameters from
    
    Returns:
        Callable: Function that imports and runs the module when called with a session
    
    Note:
        The module's main() function must accept a 'session' parameter
    """
    def func(session: Session) -> str:
        import_dir = sys._xoptions.get("snowflake_import_directory")
        # Add the name of the wheel file to the system path
        sys.path.append(import_dir + 'ml_utils-0.0.1-py3-none-any.whl')

        ctx = TaskContext(session)
        # Get parameters as dictionary for **kwargs
        params = _get_return_vals(task_context=ctx, return_from_tasks=return_from_tasks)

        # Dynamically import and execute the module's main function
        mod_name = os.path.splitext(filename)[0]
        module = importlib.import_module(mod_name)
        results = module.main(session=session, **params)

        # Store results for downstream tasks
        results = results if results else ""
        ctx.set_return_value(results)
        return results

    return func

def _get_task_definition(
    session: Session,
    file: str, 
    project_name: str,
    mljob: bool,
    return_from_tasks: list = []
) -> Callable:
    """
    Create an appropriate task function based on file type and execution mode.
    
    Routes to the correct runner based on file extension and configuration:
    - .ipynb files: Deploy as Snowflake Notebook and return notebook runner
    - .py files with mljob=True: Return ML Job runner for container execution
    - .py files with mljob=False: Return function runner for direct Python execution
    
    Args:
        session (Session): Active Snowflake session
        file (str): Filename with extension (.py or .ipynb)
        project_name (str): Project name for namespace/path resolution
        mljob (bool): If True, run Python scripts as ML jobs; if False, run directly
        return_from_tasks (list): Predecessor task names for parameter passing
    
    Returns:
        Callable: Task function ready to be used in a DAG definition
    
    Raises:
        ValueError: If file extension is not .py or .ipynb
    """
    filename, filetype = os.path.splitext(file)
    
    if filetype == ".ipynb":
        # Deploy notebook to Snowflake and create runner
        fully_qualified_name = _deploy_notebook(
            session=session,
            project_name=project_name,
        )
        task_func = _get_notebook_sql(
            session=session,
            fully_qualified_name=fully_qualified_name,
            notebook_file=file,
            return_from_tasks=return_from_tasks,
        )
    elif filetype == ".py":
        if mljob:
            # Use ML Job for container-based execution
            task_func = _get_mljob_runner(
                filename=file,
                project_name=project_name,
                return_from_tasks=return_from_tasks,
            )
        else:
            # Use direct Python function execution
            task_func = _get_func_runner(
                filename=file,
                return_from_tasks=return_from_tasks,
            )
    else:
        raise ValueError("Filetype must be py or ipynb")
    
    return task_func

def _validate_schedule(schedule: str) -> Union[timedelta, Cron, None]:
    """
    Parse and validate a schedule string into a Snowflake-compatible schedule object.
    
    Converts schedule strings into either a Cron object or timedelta object for DAG scheduling.
    Supports two formats:
    - CRON format: "CRON <cron_expression> <timezone>" (e.g., "CRON 0 9 * * * UTC")
    - Interval format: "<number> <HOURS|MINUTES|SECONDS>" (e.g., "5 HOURS")
    
    Args:
        schedule (str): Schedule string in CRON or interval format, or None
    
    Returns:
        Union[timedelta, Cron, None]: Parsed schedule object or None if schedule is empty
    
    Raises:
        ValueError: If schedule format is invalid
    """
    if schedule:
        sched_list = schedule.upper().split(" ")
        if sched_list[0]=="CRON":
            return Cron(" ".join(sched_list[1:-1]), sched_list[-1])
        elif sched_list[-1] in ["HOURS","MINUTES","SECONDS"]:
            return timedelta(**{sched_list[-1].lower():int(sched_list[0])})
        else:
            raise ValueError(f"Schedule {schedule} is not a valid value. Must be CRON value or HOURS, MINUTES, or SECONDS.")
    else:
        return schedule

def _validate_dags(dags: list[dict]) -> list[dict]:
    """
    Validate and normalize DAG configuration from project config file.
    
    Processes DAG configuration dictionaries to ensure required fields exist,
    normalize field types, set appropriate defaults, and parse schedule strings
    into Snowflake-compatible schedule objects.
    
    Args:
        dags (list[dict]): List of DAG configurations from config file
    
    Returns:
        list[dict]: Validated and normalized DAG configurations with structure:
            - name (str): DAG name
            - schedule (Union[Cron, timedelta, None]): Parsed schedule object
            - tasks (list): List of validated task configurations
            - conda_packages (list): Additional Conda packages to install
    
    Raises:
        ValueError: If 'final' or 'mljob' fields are not boolean, or if schedule format is invalid
    """
    valid_list = []
    for dag_config in dags:
        valid_dag = dict(
            name=dag_config["name"],
            schedule=dag_config.get("schedule",None),
            tasks=[],
            conda_packages=dag_config.get("conda_packages",[])
        )

        valid_dag["schedule"] = _validate_schedule(valid_dag["schedule"]) 
        
        for task_config in dag_config["tasks"]:
            # Extract and normalize task configuration
            valid_dict = dict(
                name=task_config["name"],
                file=task_config["file"],
                dep=task_config.get("dep",[]),  # Dependencies (predecessor tasks)
                final=task_config.get("final",False),  # Is this a finalizer task?
                mljob=task_config.get("mljob",False)  # Run as ML job?
            )
            
            # Ensure dependencies is always a list
            if not isinstance(valid_dict["dep"],list):
                valid_dict["dep"] = [valid_dict["dep"]]
            
            # Validate boolean fields
            if not isinstance(valid_dict["final"],bool):
                raise ValueError("Config 'final' must be True or False")
            if not isinstance(valid_dict["mljob"],bool):
                raise ValueError("Config 'mljob' must be True or False")
            
            valid_dag["tasks"].append(valid_dict)
        valid_list.append(valid_dag)
    
    return valid_list

def create_dag(
        session: Session,
        project_name: str, 
        dag_name: str,
        schedule: Union[timedelta, Cron, None],
        tasks: list,
        imports: list = [],
        packages: list = [],
    ) -> DAG:
    """
    Create and configure a Snowflake DAG with tasks and dependencies.
    
    Constructs a DAG by creating task definitions for each configured task,
    setting up dependencies between tasks, and configuring the DAG with
    required packages and imports. The DAG name is prefixed with the project
    name to ensure uniqueness across projects.
    
    Args:
        session (Session): Active Snowflake session
        project_name (str): Project name for task resolution and DAG name prefix
        dag_name (str): Base name for the DAG (will be prefixed with project_name)
        schedule (Union[timedelta, Cron, None]): Schedule object from _validate_schedule
        tasks (list): List of task configurations with 'name', 'file', 'dep', 'final', 'mljob'
        imports (list): Stage paths for all project files and dependencies
        packages (list): Conda/pip packages to install in DAG environment
    
    Returns:
        DAG: Configured Snowflake DAG object ready for deployment
    """
    with DAG(
        name=f"{project_name}_{dag_name}",
        schedule=schedule, 
        warehouse=WAREHOUSE, 
        stage_location=JOB_STAGE,
        packages=packages,
        imports=imports,
    ) as dag:
        task_ref = {}  # Track created tasks for dependency resolution
        
        for t in tasks:
            # Create appropriate task function based on file type and config
            task_func = _get_task_definition(
                session=session,
                file=t['file'],
                project_name=project_name,
                mljob=t["mljob"],
                return_from_tasks=t["dep"],
            )
            
            # Create DAG task with the function
            task = DAGTask(
                name=t["name"],
                definition=task_func,
                is_finalizer=t["final"],  # Finalizer tasks run even if predecessors fail
            )
            
            # Set up task dependencies (predecessor >> successor)
            if t["dep"]:
                for d in t["dep"]:
                    task_ref[d] >> task
            
            # Store task reference for downstream dependencies
            task_ref[t["name"]] = task
    
    return dag

def _validate_compute_resources(resource_parameters: dict) -> dict:
    """
    Validate and normalize compute resource parameters for warehouse and compute pool creation.
    
    Filters input parameters against Snowflake's allowed parameter names for warehouses
    and compute pools, merging them with sensible defaults. This ensures only valid
    parameters are passed to CREATE WAREHOUSE and CREATE COMPUTE POOL SQL statements,
    preventing SQL errors from invalid parameter names.
    
    Args:
        resource_parameters (dict): Configuration dictionary that may contain:
            - warehouse (dict): Optional warehouse parameters (e.g., WAREHOUSE_SIZE, AUTO_SUSPEND)
            - compute_pool (dict): Optional compute pool parameters (e.g., MIN_NODES, INSTANCE_FAMILY)
    
    Returns:
        dict: Validated parameters with structure:
            {
                "warehouse": {<validated_warehouse_params>},
                "compute_pool": {
                    "INSTANCE_FAMILY": "CPU_X64_XS",  # Default
                    "MIN_NODES": 1,                    # Default
                    "MAX_NODES": 1,                    # Default
                    <additional_validated_params>
                }
            }
    
    Note:
        - Parameter keys are case-insensitive (converted to uppercase for validation)
        - Invalid parameter names are silently ignored
        - Warehouse parameters default to empty dict if none provided
        - Compute pool always gets default values for INSTANCE_FAMILY, MIN_NODES, MAX_NODES
    
    Example:
        >>> params = {
        ...     "warehouse": {"WAREHOUSE_SIZE": "SMALL", "AUTO_SUSPEND": 300},
        ...     "compute_pool": {"MIN_NODES": 2}
        ... }
        >>> _validate_compute_resources(params)
        {
            "warehouse": {"WAREHOUSE_SIZE": "SMALL", "AUTO_SUSPEND": 300},
            "compute_pool": {"INSTANCE_FAMILY": "CPU_X64_XS", "MIN_NODES": 2, "MAX_NODES": 1}
        }
    """
    # Initialize valid parameters with defaults
    # Warehouse starts empty (Snowflake will use system defaults)
    # Compute pool gets minimal configuration defaults
    valid_params = {
        "warehouse":{},
        "compute_pool":{
            "INSTANCE_FAMILY":"CPU_X64_XS",  # Smallest CPU instance family
            "MIN_NODES":1,                    # Single node minimum
            "MAX_NODES":1,                    # No auto-scaling by default
        }
    }
    
    # Define allowed parameter names for each resource type
    # Based on Snowflake SQL reference documentation for CREATE WAREHOUSE and CREATE COMPUTE POOL
    valid_keys = {
        "warehouse":[
            "WAREHOUSE_SIZE","WAREHOUSE_TYPE","RESOURCE_CONSTRAINT","MAX_CLUSTER_COUNT","MIN_CLUSTER_COUNT","SCALING_POLICY","AUTO_SUSPEND",
            "AUTO_RESUME","INITIALLY_SUSPENDED","RESOURCE_MONITOR","COMMENT","ENABLE_QUERY_ACCELERATION","QUERY_ACCELERATION_MAX_SCALE_FACTOR",
            "MAX_CONCURRENCY_LEVEL","STATEMENT_QUEUED_TIMEOUT_IN_SECONDS","STATEMENT_TIMEOUT_IN_SECONDS"
        ],
        "compute_pool":[
            "MIN_NODES","MAX_NODES","INSTANCE_FAMILY","AUTO_RESUME","INITIALLY_SUSPENDED","AUTO_SUSPEND_SECS","COMMENT","PLACEMENT_GROUP"
        ]
    }
    
    # Validate warehouse parameters if provided
    warehouse = resource_parameters.get("warehouse",None)
    if warehouse:
        for k,v in warehouse.items():
            # Convert key to uppercase for case-insensitive validation
            if k.upper() in valid_keys["warehouse"]:
                valid_params["warehouse"][k] = v
                # Note: Invalid keys are silently ignored
    
    # Validate compute pool parameters if provided
    compute_pool = resource_parameters.get("compute_pool",None)
    if compute_pool:
        for k,v in compute_pool.items():
            # Convert key to uppercase for case-insensitive validation
            if k.upper() in valid_keys["compute_pool"]:
                # User-provided values override defaults
                valid_params["compute_pool"][k] = v
                # Note: Invalid keys are silently ignored
    
    return valid_params


def _create_compute_resources(session: Session, project_name: str, compute_resource_params: dict) -> None:
    """
    Create project-specific compute resources (warehouse and compute pool).
    
    Dynamically creates a Snowflake warehouse and compute pool for the project,
    setting global variables WAREHOUSE and COMPUTE_POOL for use by deployment
    functions. Resource names are prefixed with the sanitized project name and
    environment to ensure isolation between projects and environments.
    Grants appropriate privileges based on the current environment.
    
    Args:
        session (Session): Active Snowflake session
        project_name (str): Project name (will be sanitized for Snowflake identifiers)
        compute_resource_params (dict): Validated compute resource parameters
    
    Side Effects:
        Sets global variables WAREHOUSE and COMPUTE_POOL
        Grants privileges to ACCOUNTADMIN role
    
    Note:
        Non-alphanumeric characters in project_name are replaced with underscores
    """
    project_name = re.sub(r"[^A-Z0-9]","_", project_name.upper())

    global WAREHOUSE
    WAREHOUSE = f"ML_PIPELINE_{project_name}_{ENVIRONMENT}_WH"
    wh_sql = " ".join([f"{k} = {v}" for k,v in compute_resource_params["warehouse"].items()])
    
    # Create warehouse if it doesn't exist (preserves existing warehouse and its usage history)
    session.sql(f"""
        CREATE WAREHOUSE IF NOT EXISTS {WAREHOUSE};
    """).collect()
    
    # Update warehouse properties if any are specified
    if wh_sql:
        session.sql(f"""
            ALTER WAREHOUSE {WAREHOUSE} SET {wh_sql};
        """).collect()
    
    # Grant privileges on warehouse based on environment
    _grant_privileges(session, "warehouse", WAREHOUSE)

    global COMPUTE_POOL
    COMPUTE_POOL = f"ML_PIPELINE_{project_name}_{ENVIRONMENT}_COMPUTE"
    cp_sql = " ".join([f"{k} = {v}" for k,v in compute_resource_params["compute_pool"].items()])
        
    session.sql(f"""
        CREATE COMPUTE POOL IF NOT EXISTS {COMPUTE_POOL} {cp_sql};
    """).collect()

    if cp_sql:
        session.sql(f"""
            ALTER COMPUTE POOL {COMPUTE_POOL} SET {cp_sql};
        """).collect()
    
    # Grant privileges on compute pool based on environment
    _grant_privileges(session, "compute_pool", COMPUTE_POOL)

def _deprecate_dags(session: Session, project_name: str, deployed_dags: list[DAG]) -> None:
    """
    Deprecate DAGs that are not in config file.
    
    Args:
        session: Snowflake session
        project_name: Project name
        deployed_dags: List of deployed DAG objects
    """

    project_dags = session.sql(f"""
        SHOW TASKS LIKE '{project_name}%' IN SCHEMA {session.get_current_database()}.{session.get_current_schema()}
    """).collect()

    deployed_dag_names = [d.name.upper() for d in deployed_dags]

    for d in project_dags:
        if d.name.split('$')[0] not in deployed_dag_names:
            session.sql(f"""
                ALTER TASK {d.name} SUSPEND; 
            """).collect()
            session.sql(f"""
                ALTER TASK {d.name} SET COMMENT = 'DEPRECATED: This DAG is no longer in the project configuration file.';
            """).collect()
            print(f"Suspended deprecated DAG {d.name}")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        "Deploy an ML project in Snowflake",
        description="Create and deploy machine learning DAGs based on project configuration files.",
    )

    parser.add_argument(
        "project_name",
        help="The name of the project subdirectory to be deployed",
    )

    parser.add_argument(
        "--run-dag",
        action="store_true",
        default=False,
        help="Execute all tasks immediately after deployment. If not specified, the tasks will only be deployed and can be run manually or according to the schedule.",
    )

    args = parser.parse_args()

    # Load project configuration from YAML file and validate DAG definitions
    config = yaml.safe_load(open(f"projects/{args.project_name}/config.yml","r"))
    config['active'] = config.get("active",False)
    if not config['active']:
        print(f"Project {args.project_name} is not active. Skipping deployment.")
        exit(0)
    
    dags = _validate_dags(config['deploy']['DAGS'])
    compute_resource_params = _validate_compute_resources(config['deploy'])

    # Initialize Snowflake session with configured connection
    # Use connection name if provided for local development, otherwise use user, password, and account
    session = get_session()
    session.use_role(ROLE_NAME)
    session.use_database(DB_NAME)
    session.use_schema(SCHEMA_NAME)

    # Create project-specific warehouse and compute pool (sets global WAREHOUSE and COMPUTE_POOL)
    _create_compute_resources(session, args.project_name, compute_resource_params)
    session.use_warehouse(WAREHOUSE)
    
    # Grant schema-level privileges based on environment
    schema_name = f"{session.get_current_database()}.{session.get_current_schema()}"
    _grant_privileges(session, "schema", schema_name)

    # Upload project files and dependencies to BUILD_STAGE and JOB_STAGE
    staged_files = stage_directory(session, args.project_name)

    # Initialize Snowflake API objects for DAG operations
    api_root = Root(session)
    db = api_root.databases[session.get_current_database()]
    schema = db.schemas[session.get_current_schema()]
    dag_op = DAGOperation(schema)

    # Deploy each DAG defined in the project configuration
    deployed_dags = []
    for d in dags:
        
        # Create DAG with all configured tasks, dependencies, and packages
        dag = create_dag(
            session=session,
            project_name=args.project_name,
            dag_name=d["name"],
            schedule=d["schedule"],
            tasks=d["tasks"],
            imports=staged_files,  # All project files and ml_utils wheel
            packages=["snowflake-snowpark-python","snowflake-ml-python"]+d["conda_packages"],
        )

        # Deploy to Snowflake (replaces existing DAG with same name)
        dag_op.deploy(dag, mode=CreateMode.or_replace)
        deployed_dags.append(dag)
        
        # Grant privileges on all tasks in the DAG
        # Root task has the same name as the DAG
        _grant_privileges(session, "task", f"{schema_name}.{dag.name}")
        # Child tasks have names in format DAG_NAME$TASK_NAME
        for task in dag.tasks:
            _grant_privileges(session, "task", f"{schema_name}.{dag.name}${task.name}")
    

    # Optionally execute DAGs immediately for validation/testing (CI/CD use)
    if args.run_dag:
        for d in deployed_dags:
            dag_op.run(d)
            result = _wait_for_run_to_complete(session, d)
            if result != "SUCCEEDED":
                raise Exception(f"DAG {d.name} failed with result {result}")

    _deprecate_dags(session, args.project_name, deployed_dags)

    # Suspend DAGs in non-PROD environments to prevent accidental execution
    if ENVIRONMENT != "PROD":
        for d in deployed_dags:
            session.sql(f"ALTER TASK {session.get_current_database()}.{session.get_current_schema()}.{d.name} SUSPEND;").collect()

    session.close()
