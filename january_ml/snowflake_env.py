import os
from snowflake.snowpark.session import Session

CONNECTION = os.getenv("SNOWFLAKE_CONNECTION")
ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
USER = os.getenv("SNOWFLAKE_USER")
PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
ENVIRONMENT = os.getenv("SNOWFLAKE_ENVIRONMENT")


def get_session():
    if ENVIRONMENT:
        connect_configs = {"connection_name": CONNECTION} if CONNECTION else {
                "user": USER,
                "password": PASSWORD,
                "account": ACCOUNT,
            }
        connect_configs['database'] = f"ML_COLLAB_{ENVIRONMENT}_DB"
        session = Session.builder.configs(connect_configs).create()
    else:
        session = Session.builder.getOrCreate()

    return session

def get_feature_schema(session: Session) -> str:
    return "SANDBOX" if session.get_current_database() == '"ML_COLLAB_DEV_DB"' else "FEATURES"

def get_model_schema(session: Session) -> str:
    return "SANDBOX" if session.get_current_database() == '"ML_COLLAB_DEV_DB"' else "MODELS"

def get_build_stage(session: Session) -> str:
    DB_NAME = session.get_current_database().replace('"','')
    return f"{DB_NAME}.{get_model_schema(session)}.BUILD_STAGE"


def get_job_stage(session: Session) -> str:
    DB_NAME = session.get_current_database().replace('"','')
    return f"{DB_NAME}.{get_model_schema(session)}.JOB_STAGE"