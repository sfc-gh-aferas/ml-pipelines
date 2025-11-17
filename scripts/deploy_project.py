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
    Requires Snowflake connection configuration via january_ml.constants module.
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
from january_ml.constants import (
    CONNECTION,
    DB_NAME,
    SCHEMA_NAME,
    ROLE_NAME,
    ENVIRONMENT,
    #GIT_STAGE,
    BUILD_STAGE,
    JOB_STAGE,
    # BRANCH,
)

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
    recent_runs = session.sql(
        f"""
        select run_id
            from table({DB_NAME}.information_schema.current_task_graphs(
                root_task_name => '{dag.name.upper()}'
            ))
            where database_name = '{DB_NAME}'
            and schema_name = '{SCHEMA_NAME}'
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
            select state
                from table({DB_NAME}.information_schema.complete_task_graphs(
                    root_task_name=>'{dag.name.upper()}'
                ))
                where database_name = '{DB_NAME}'
                and schema_name = '{SCHEMA_NAME}'
                and run_id = {run_id};
            """,
        ).collect()

        if len(result) > 0:
            dag_result = result[0][0]
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
    
    Creates two Snowflake stages (BUILD_STAGE, JOB_STAGE) if they don't exist,
    removes any previous project files from these stages, and uploads all files
    from the project directory along with the january_ml package wheel.
    
    Args:
        session (Session): Active Snowflake session
        project_dir (str): Name of the project subdirectory under 'projects/'
    
    Returns:
        list[str]: List of stage paths for all uploaded files (e.g., '@BUILD_STAGE/project/file.py')
    """
    # Create stages and remove previous project files
    session.sql(f"CREATE STAGE IF NOT EXISTS {BUILD_STAGE}").collect()
    session.sql(f"REMOVE @{BUILD_STAGE}/{project_dir}").collect()
    session.sql(f"CREATE STAGE IF NOT EXISTS {JOB_STAGE}").collect()
    session.sql(f"REMOVE @{JOB_STAGE}/{project_dir}").collect()

    # Upload all non-directory files from the project directory
    staged_files = []
    for f in os.listdir(f"projects/{project_dir}"):
        filename = f"projects/{project_dir}/{f}"
        if not os.path.isdir(filename):
            result = session.file.put(filename,f"{BUILD_STAGE}/{project_dir}",overwrite=True, auto_compress=False)
            staged_files.append(f"@{BUILD_STAGE}/{project_dir}/{result[0].target}")
    
    # Upload the january_ml package wheel file
    result = session.file.put(
        "dist/january_ml-0.0.1-py3-none-any.whl",
        f"{BUILD_STAGE}/{project_dir}/dist", 
        overwrite=True, 
        auto_compress=False
    )
    staged_files.append(f"@{BUILD_STAGE}/{project_dir}/dist/{result[0].target}")
    print(f"{project_dir} uploaded to {BUILD_STAGE}")
    return staged_files

def _deploy_notebook(session: Session, notebook_file: str, project_name: str) -> str:
    """
    Deploy a Jupyter notebook to Snowflake as a Snowflake Notebook.
    
    Creates a Snowflake Notebook from a .ipynb file in BUILD_STAGE with configured
    runtime, warehouse, and compute pool. Activates a live version for execution.
    Uses dynamically created WAREHOUSE and COMPUTE_POOL global variables.
    
    Args:
        session (Session): Active Snowflake session
        notebook_file (str): Name of the .ipynb file (e.g., 'training.ipynb')
        project_name (str): Project name used as namespace prefix
    
    Returns:
        str: Fully qualified name of the deployed notebook (DB.SCHEMA.PROJECT__NOTEBOOK)
    
    Note:
        TODO: Add support for Git branch-based deployment
        TODO: Make compute pool selection configurable per notebook
        TODO: Add error handling logic for notebook creation failures
    """
    # Create fully qualified notebook name with project namespace
    notebook_name = notebook_file.replace(".ipynb","")
    fully_qualified_name = f"{DB_NAME}.{SCHEMA_NAME}.{project_name}__{notebook_name}"
    
    # Create notebook with runtime configuration
    nb_sql = f"""
        CREATE OR REPLACE NOTEBOOK {fully_qualified_name}
        FROM @{BUILD_STAGE}/{project_name}
        MAIN_FILE = '{notebook_file}'
        QUERY_WAREHOUSE = {WAREHOUSE}
        RUNTIME_NAME = 'SYSTEM$BASIC_RUNTIME'
        COMPUTE_POOL = {COMPUTE_POOL}
        IDLE_AUTO_SHUTDOWN_TIME_SECONDS = 3600;
    """
    results = session.sql(nb_sql).collect()

    # Activate a live version of the notebook for execution
    alter_sql = f"""ALTER NOTEBOOK {fully_qualified_name} ADD LIVE VERSION FROM LAST;"""
    session.sql(alter_sql).collect()
    # TODO: logic to handle failure

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
            val = json.loads(result)
            if script_args:
                kw += [i for k,v in val.items() for i in ("--"+str(k),str(v))]
            else:
                kw.update(val)         
    return kw

def _get_notebook_runner(fully_qualified_name: str, return_from_tasks: list = []) -> Callable:
    """
    Create a task function that executes a deployed Snowflake Notebook.
    
    Returns a callable function that can be used as a DAG task definition.
    The function executes the notebook with parameters from predecessor tasks.
    Handles cases where no parameters are provided (executes with empty params).
    
    Args:
        fully_qualified_name (str): Full notebook name (DB.SCHEMA.NOTEBOOK_NAME)
        return_from_tasks (list): Task names to retrieve parameters from
    
    Returns:
        Callable: Function that executes the notebook when called with a session
    """
    def nb_func(session: Session) -> str:
        ctx = TaskContext(session)
        # Get parameters from predecessor tasks and format as SQL arguments
        params = _get_return_vals(task_context=ctx, return_from_tasks=return_from_tasks, script_args=True)
        params = "'"+"','".join(params)+"'" if params else ""
        return session.sql(f"EXECUTE NOTEBOOK {fully_qualified_name}({params});").collect()
        
    return nb_func

def _get_mljob_runner(filename: str, project_name: str, return_from_tasks: list = []) -> Callable:
    """
    Create a task function that submits and runs a Python script as a Snowflake ML Job.
    
    Returns a callable that submits a Python script from BUILD_STAGE as a Snowflake ML Job
    with dependencies and parameters from predecessor tasks. The job runs on the dynamically
    created COMPUTE_POOL with isolated container execution. Includes pip requirements from
    the project's pip-requirements.txt and the january_ml package wheel.
    
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
            pip_requirements=["-r ../app/pip-requirements.txt"],
            imports=[f"@{BUILD_STAGE}/{project_name}/dist"]  # Include january_ml package
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
    that don't require ML Job submission. Adds the january_ml wheel to sys.path
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
        sys.path.append(import_dir + 'january_ml-0.0.1-py3-none-any.whl')

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
            notebook_file=file, 
            project_name=project_name,
        )
        task_func = _get_notebook_runner(
            fully_qualified_name=fully_qualified_name,
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

def _create_compute_resources(session: Session, project_name: str) -> None:
    """
    Create project-specific compute resources (warehouse and compute pool).
    
    Dynamically creates a Snowflake warehouse and compute pool for the project,
    setting global variables WAREHOUSE and COMPUTE_POOL for use by deployment
    functions. Resource names are prefixed with the sanitized project name and
    environment to ensure isolation between projects and environments.
    
    Args:
        session (Session): Active Snowflake session
        project_name (str): Project name (will be sanitized for Snowflake identifiers)
    
    Side Effects:
        Sets global variables WAREHOUSE and COMPUTE_POOL
    
    Note:
        Non-alphanumeric characters in project_name are replaced with underscores
    """
    project_name = re.sub(r"[^A-Z0-9]","_", project_name.upper())

    global WAREHOUSE
    WAREHOUSE = f"{project_name}_{ENVIRONMENT}_WH"
    session.sql(f"""
        CREATE WAREHOUSE IF NOT EXISTS {WAREHOUSE}
            WAREHOUSE_SIZE = SMALL;
    """).collect()

    global COMPUTE_POOL
    COMPUTE_POOL = f"{project_name}_{ENVIRONMENT}_COMPUTE"
    session.sql(f"""
        CREATE COMPUTE POOL IF NOT EXISTS {COMPUTE_POOL}
            MIN_NODES = 1
            MAX_NODES = 1
            INSTANCE_FAMILY = CPU_X64_M
    """).collect()

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
    dags = _validate_dags(config['deploy']['DAGS'])

    # Initialize Snowflake session with configured connection
    session = Session.builder.config("connection_name",CONNECTION).getOrCreate()
    session.use_role(ROLE_NAME)
    session.use_database(DB_NAME)
    session.use_schema(SCHEMA_NAME)

    # Create project-specific warehouse and compute pool (sets global WAREHOUSE and COMPUTE_POOL)
    _create_compute_resources(session, args.project_name)
    session.use_warehouse(WAREHOUSE)

    # Upload project files and dependencies to BUILD_STAGE and JOB_STAGE
    staged_files = stage_directory(session, args.project_name)

    # Initialize Snowflake API objects for DAG operations
    api_root = Root(session)
    db = api_root.databases[DB_NAME]
    schema = db.schemas[SCHEMA_NAME]
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
            imports=staged_files,  # All project files and january_ml wheel
            packages=["snowflake-snowpark-python","snowflake-ml-python"]+d["conda_packages"],
        )

        # Deploy to Snowflake (replaces existing DAG with same name)
        dag_op.deploy(dag, mode=CreateMode.or_replace)
        deployed_dags.append(dag)
    
    # Optionally execute DAGs immediately for validation/testing (CI/CD use)
    if args.run_dag:
        for d in deployed_dags:
            dag_op.run(d)
            result = _wait_for_run_to_complete(session, d)
            if result != "SUCCEEDED":
                raise Exception(f"DAG {d.name} failed with result {result}")

    session.close()
