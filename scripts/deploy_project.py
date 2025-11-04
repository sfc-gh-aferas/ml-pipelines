# Takes project name as arg
# env var STAGE/PROD

# Git stage, pull

# IF NOTEBOOKS
    # create notebooks based on current branch, container runtime
    # schedule task to execute notebook

# IF PY SCRIPTS
    # create task with MLJob submit_from_stage on current git branch

from january_ml.utils import load_config
import os
import time
from snowflake.snowpark.session import Session
from snowflake.core import CreateMode, Root
from snowflake.ml.jobs import submit_from_stage
from snowflake.core.task.dagv1 import DAG, DAGTask, DAGOperation
from january_ml.constants import (
    CONNECTION,
    DB_NAME,
    SCHEMA_NAME,
    ROLE_NAME,
    WAREHOUSE,
    COMPUTE_POOL,
    GIT_STAGE,
    JOB_STAGE,
    PACKAGE_STAGE,
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

# temporarily using this approach for testing in absence of access git repo
def stage_directory(session: Session, project_dir: str):

    session.sql(f"CREATE OR REPLACE STAGE {GIT_STAGE}").collect()
    session.sql(f"CREATE OR REPLACE STAGE {JOB_STAGE}").collect()
    session.sql(f"CREATE OR REPLACE STAGE PACKAGE_STAGE").collect()

    for f in os.listdir("projects/"+project_dir):
        filename = "projects/"+project_dir+"/"+f
        if not os.path.isdir(filename):
            session.file.put(filename,GIT_STAGE+"/"+project_dir,overwrite=True, auto_compress=False)
    session.file.put('dist/january_ml-0.0.1-py3-none-any.whl',f"{PACKAGE_STAGE}/dist", overwrite=True, auto_compress=False)
    print(f"{project_dir} uploaded to {GIT_STAGE}")

def _deploy_notebook(session: Session, notebook_name: str, project_name: str) -> str:
    
    fully_qualified_name = f"{DB_NAME}.{SCHEMA_NAME}.{project_name}__{notebook_name}"

    # TODO: fix for git branch
    # TODO: fix for compute pool
    nb_sql = f"""
        CREATE OR REPLACE NOTEBOOK {fully_qualified_name}
        FROM @{GIT_STAGE}/{project_name}
        MAIN_FILE = '{notebook_name}.ipynb'
        QUERY_WAREHOUSE = {WAREHOUSE}
        RUNTIME_NAME = 'SYSTEM$BASIC_RUNTIME'
        COMPUTE_POOL = {COMPUTE_POOL}
        IDLE_AUTO_SHUTDOWN_TIME_SECONDS = 3600;
    """
    results = session.sql(nb_sql).collect()

    alter_sql = f"""ALTER NOTEBOOK {fully_qualified_name} ADD LIVE VERSION FROM LAST;"""
    session.sql(alter_sql).collect()
    # logic to handle failure

    print(f"Successfully deployed notebook {fully_qualified_name}")

    return f"EXECUTE NOTEBOOK {fully_qualified_name}();"

def _deploy_mljob(session: Session, filename: str, project_name: str):

    def job_func(session: Session) -> str:
        stage_path = f"@{GIT_STAGE}/{project_name}"
        job = submit_from_stage(
            source=stage_path,
            compute_pool=COMPUTE_POOL,
            entrypoint=f"{filename}",
            stage_name=JOB_STAGE,
            session=session,
            pip_requirements=["-r ../app/project-requirements.txt"],
            imports=[f"@{PACKAGE_STAGE}/dist"]
        )
        return job.result()

    return job_func

def create_dag(
        project_name: str, 
        dag_name: str,
        schedule: str,
        task_files: list
    ) -> DAG:

    with DAG(
        name=dag_name,
        schedule=schedule, 
        warehouse=WAREHOUSE, 
        stage_location=JOB_STAGE,
        packages=["snowflake-ml-python","snowflake-snowpark-python"],
    ) as dag:
        prev_task = None
        for f in task_files:
            name, filetype = ".".join(f.split(".")[:-1]), f.split(".")[-1]
            if filetype == "ipynb":
                nb_deploy = _deploy_notebook(
                    session=session,
                    notebook_name=name, 
                    project_name=project_name
                )
                task = DAGTask(
                    name=name,
                    definition=nb_deploy,
                )
            elif filetype == "py":
                mljob = _deploy_mljob(
                    session=session,
                    filename=f,
                    project_name=project_name
                )
                task = DAGTask(
                    name=name,
                    definition=mljob,
                )
            # else:
                # needs to be py or ipynb
            if prev_task:
                prev_task >> task
            prev_task=task
    return dag

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

    config = load_config(args.project_name)
    dags = config['deploy']['DAGS']

    session = Session.builder.config("connection_name",CONNECTION).getOrCreate()
    # session level query tags?
    session.use_role(ROLE_NAME)
    session.use_warehouse(WAREHOUSE)
    session.use_database(DB_NAME)
    session.use_schema(SCHEMA_NAME)

    #temporary
    stage_directory(session, args.project_name)

    api_root = Root(session)
    db = api_root.databases[DB_NAME]
    schema = db.schemas[SCHEMA_NAME]
    dag_op = DAGOperation(schema)

    deployed_dags = []
    for d in dags:
        dag = create_dag(
            project_name=args.project_name,
            dag_name=d["name"],
            schedule=d["schedule"],
            task_files=d["task_files"],
        )
        dag_op.deploy(dag, mode=CreateMode.or_replace)
        deployed_dags.append(dag)
            
    if args.run_dag:
        for d in deployed_dags:
            dag_op.run(d)
            result = _wait_for_run_to_complete(session, d)
            if result != "SUCCEEDED":
                raise Exception(f"DAG failed with result {result}")

    session.close()
