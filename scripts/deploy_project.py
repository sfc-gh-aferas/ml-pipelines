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
from snowflake.snowpark.session import Session
from snowflake.core import Root
from snowflake.core.task import Task
from constants import (
    CONNECTION,
    DB_NAME,
    SCHEMA_NAME,
    ROLE_NAME,
    WAREHOUSE,
    #COMPUTE_POOL,
    GIT_STAGE,
    # BRANCH,
)
# temporarily using this approach for testing in absence of access git repo
def stage_directory(session: Session, project_dir: str):

    stage_name = f"{DB_NAME}.{SCHEMA_NAME}.{GIT_STAGE}"
    session.sql(f"CREATE STAGE IF NOT EXISTS {stage_name}").collect()

    for f in os.listdir(project_dir):
        filename = project_dir+"/"+f
        if not os.path.isdir(filename):
            session.file.put(filename,stage_name+'/'+project_dir,overwrite=True, auto_compress=False)
    session.file.put('dist/january_ml-0.0.1-py3-none-any.whl',stage_name, overwrite=True, auto_compress=False)
    print(f"{project_dir} uploaded to {stage_name}")

def _schedule_notebook(session: Session, fully_qualified_name: str, schedule: str):

    # validate schedule (see e2e repo cli_utils.validate_schedule)
    task_name = f"{fully_qualified_name.split('.')[-1]}_TASK"

    root = Root(session)
    task = Task(
        name=task_name, 
        definition=f"EXECUTE NOTEBOOOK {fully_qualified_name}();", 
        schedule=schedule,
        warehouse=WAREHOUSE,
    )
    # create a task collections objects, specifying the database and schema
    tasks = root.databases[DB_NAME].schemas[SCHEMA_NAME].tasks
    # create the task in Snowflake
    tasks.create(task)

    #logic to handle failure
    print(f"Successfully scheduled task {task_name}")
    return tasks[task_name]


def _deploy_notebook(session: Session, notebook_name: str, project_name: str) -> str:
    
    fully_qualified_name = f"{DB_NAME}.{SCHEMA_NAME}.{project_name}__{notebook_name}"

    # TODO: fix for git branch
    # TODO: fix for compute pool
    nb_sql = f"""
        CREATE NOTEBOOK IF NOT EXISTS {fully_qualified_name}
        FROM @{GIT_STAGE}/{project_name}
        MAIN_FILE = '{notebook_name}.ipynb'
        QUERY_WAREHOUSE = {WAREHOUSE}
        RUNTIME_NAME = 'SYSTEM$BASIC_RUNTIME' 
        IDLE_AUTO_SHUTDOWN_TIME_SECONDS = 3600
    """

    results = session.sql(nb_sql).collect()
    # logic to handle failure

    print(f"Successfully deployed notebook {fully_qualified_name}")

    return fully_qualified_name

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
        "--run-tasks",
        action="store_true",
        default=False,
        help="Execute all tasks immediately after deployment. If not specified, the tasks will only be deployed and can be run manually or according to the schedule.",
    )

    args = parser.parse_args()

    config = load_config(f"{args.project_name}/config.yml")
    pipelines = config['deploy']['pipelines']

    session = Session.builder.config("connection_name",CONNECTION).getOrCreate()
    # session level query tags?
    session.use_role(ROLE_NAME)
    session.use_warehouse(WAREHOUSE)
    session.use_database(DB_NAME)
    session.use_schema(SCHEMA_NAME)

    #temporary
    stage_directory(session, args.project_name)

    scheduled_tasks = []

    for p in pipelines:
        name, filetype = ".".join(p["file"].split(".")[:-1]), p["file"].split(".")[-1]
        if filetype == "ipynb":
            nb_deploy = _deploy_notebook(
                session=session,
                notebook_name=name, 
                project_name=args.project_name
            )
            task = _schedule_notebook(
                session=session,
                fully_qualified_name=nb_deploy,
                schedule=p['schedule']
            )
            scheduled_tasks.append(task)
        # elif filetype == "py":
        # else:
            # needs to be py or ipynb
    
    if args.run_tasks:
        for t in scheduled_tasks:
            result = t.execute() 
            # how to handle failure

    session.close()
