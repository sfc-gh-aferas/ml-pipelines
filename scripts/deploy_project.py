# Takes project name as arg
# env var STAGE/PROD

# Git stage, pull

# IF NOTEBOOKS
    # create notebooks based on current branch, container runtime
    # schedule task to execute notebook

# IF PY SCRIPTS
    # create task with MLJob submit_from_stage on current git branch

import yaml
from snowflake.snowpark.session import Session
from snowflake.core import Root
from snowflake.core.task import Task

# need to get these from ci/cd env vars
# ROLE
# WAREHOUSE
# COMPUTE_POOL
# DATABASE
# GIT_STAGE
# BRANCH

def _schedule_notebook(session: Session, fully_qualified_name: str, schedule: str):

    # validate schedule (see e2e repo cli_utils.validate_schedule)
    task_name = f"{fully_qualified_name.split('.')[-1]}_TASK"

    root = Root(session)
    task = Task(name=task_name, 
                  definition=f"EXECUTE NOTEBOOOK {fully_qualified_name}();", 
                  schedule=schedule
                  warehouse=WAREHOUSE)
    # create a task collections objects, specifying the database and schema
    tasks = root.databases[DATABASE].schemas[SCHEMA].tasks
    # create the task in Snowflake
    tasks.create(task)

    #logic to handle failure
    print(f"Successfully scheduled task {task_name}")
    return tasks[task_name]


def _deploy_notebook(session: Session, notebook_name: str, project_name: str) -> str:
    
    fully_qualified_name = f"{DATABASE}.{SCHEMA}.{project_name}__{notebook_name}"
    nb_sql = """CREATE NOTEBOOK IF NOT EXISTS {fully_qualified_name}
    FROM '{GIT_STAGE}/branches/{BRANCH}/{project_name}'
    MAIN_FILE = '{notebook_name}.ipynb'
    QUERY_WAREHOUSE = {WAREHOUSE}
    RUNTIME_NAME = 'SYSTEM$BASIC_RUNTIME' 
    COMPUTE_POOL = '{COMPUTE_POOL}'
    IDLE_AUTO_SHUTDOWN_TIME_SECONDS = 3600"""

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

    config = yaml.safe_load(open(f"{args.project_name}/config.yml","r"))
    pipelines = config['deploy']['pipelines']

    session = Session.builder.getOrCreate()
    # session level query tags?

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
