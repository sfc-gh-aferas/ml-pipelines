import os
from snowflake.snowpark.session import Session
from snowflake.snowpark.context import get_active_session

CONNECTION = os.getenv("SNOWFLAKE_CONNECTION")
ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
USER = os.getenv("SNOWFLAKE_USER")
PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
ENVIRONMENT = os.getenv("SNOWFLAKE_ENVIRONMENT")

ROLE_NAME = f"ML_{ENVIRONMENT}_ROLE"
DB_NAME = f"ML_{ENVIRONMENT}_DB"
SCHEMA_NAME = F"ML_{ENVIRONMENT}_SCHEMA"

BUILD_STAGE = f"{DB_NAME}.{SCHEMA_NAME}.BUILD_STAGE"
JOB_STAGE = f"{DB_NAME}.{SCHEMA_NAME}.JOB_STAGE"


def get_session():
    try:
        session = get_active_session()
    except:
        connect_configs = {"connection_name": CONNECTION} if CONNECTION else {
            "user": USER,
            "password": PASSWORD,
            "account": ACCOUNT,
            }
        session = Session.builder.configs(connect_configs).create()
    if ENVIRONMENT:
        session.use_role(ROLE_NAME)
        session.use_database(DB_NAME)
        session.use_schema(SCHEMA_NAME)
    return session