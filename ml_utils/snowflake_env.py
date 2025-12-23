import os
from snowflake.snowpark.session import Session

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
    if ENVIRONMENT:
        connect_configs = {"connection_name": CONNECTION} if CONNECTION else {
                "user": USER,
                "password": PASSWORD,
                "account": ACCOUNT,
            }
        connect_configs['role'] = ROLE_NAME
        connect_configs['database'] = DB_NAME
        connect_configs['schema'] = SCHEMA_NAME
        session = Session.builder.configs(connect_configs).create()
    else:
        session = Session.builder.getOrCreate()

    return session