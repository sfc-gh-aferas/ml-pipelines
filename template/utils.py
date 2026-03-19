"""
This file can be used for general project utilities.
"""

from snowflake.snowpark import Session
import os
import sys
import subprocess

def install_stage_package(): -> None:
    """
    Helper function for Snowflake notebooks to get the january_ml package from the BUILD_STAGE.

    Usage: 

    from utils import get_stage_packages
    get_stage_packages()
    ! pip install ./dist/january_ml-0.0.1-py3-none-any.whl #or ! pip install -r pip-requirements.txt
    """
    session = Session.builder.getOrCreate()
    try:
        idx = sys.argv.index("--snowflake-env") + 1
        env = sys.argv[idx]
    except ValueError:
        env = "DEV"
    os.environ["SNOWFLAKE_ENVIRONMENT"] = env
    fq_schema = f"ML_PIPELINE_{env}_DB.ML_PROJECTS"
    session.file.get(f"@{fq_schema}.BUILD_STAGE/example_project/dist/ml_utils-0.0.1-py3-none-any.whl", "/tmp")
    subprocess.call([sys.executable, '-m', 'pip', 'install', "/tmp/ml_utils-0.0.1-py3-none-any.whl"])