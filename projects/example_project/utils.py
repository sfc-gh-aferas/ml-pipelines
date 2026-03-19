from snowflake.snowpark import Session
import os
import sys
import subprocess

def install_stage_package():
    session = Session.builder.getOrCreate()
    try:
        idx = sys.argv.index("--snowflake-env") + 1
        env = sys.argv[idx]
    except ValueError:
        env = "DEV"
    os.environ["SNOWFLAKE_ENVIRONMENT"] = env
    fq_schema = f"ML_{env}_DB.ML_{env}_SCHEMA"
    session.file.get(f"@{fq_schema}.BUILD_STAGE/example_project/dist/ml_utils-0.0.1-py3-none-any.whl", "/tmp")
    subprocess.call([sys.executable, '-m', 'pip', 'install', "/tmp/ml_utils-0.0.1-py3-none-any.whl"])