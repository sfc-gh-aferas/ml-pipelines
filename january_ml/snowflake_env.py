import os
from snowflake.snowpark.session import Session

CONNECTION = os.getenv("SNOWFLAKE_CONNECTION")
ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
USER = os.getenv("SNOWFLAKE_USER")
PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")

ENVIRONMENT = os.getenv("SNOWFLAKE_ENVIRONMENT") or "DEV"
DB_NAME = os.getenv("SNOWFLAKE_DATABASE") or f"ML_COLLAB_{ENVIRONMENT}_DB"


def get_session():

    connect_configs = {"connection_name": CONNECTION} if CONNECTION else {
            "user": USER,
            "password": PASSWORD,
            "account": ACCOUNT,
        }
    connect_configs['database'] = DB_NAME
    session = Session.builder.configs(connect_configs).create()

    return session

def get_feature_schema(session: Session) -> str:
    return "SHARED_WORK" if session.get_current_database() == '"ML_COLLAB_DEV_DB"' else "FEATURES"

def get_model_schema(session: Session) -> str:
    return "SHARED_WORK" if session.get_current_database() == '"ML_COLLAB_DEV_DB"' else "MODELS"

def get_build_stage(session: Session) -> str:
    return f"{session.get_current_database()}.{get_model_schema(session)}.BUILD_STAGE"


def get_job_stage(session: Session) -> str:
    return f"{session.get_current_database()}.{get_model_schema(session)}.JOB_STAGE"