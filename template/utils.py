"""
This file can be used for general project utilities.
"""

from snowflake.snowpark import Session

def get_stage_packages() -> None:
    """
    Helper function for Snowflake notebooks to get the january_ml package from the BUILD_STAGE.

    Usage: 

    from utils import get_stage_packages
    get_stage_packages()
    ! pip install ./dist/january_ml-0.0.1-py3-none-any.whl #or ! pip install -r pip-requirements.txt
    """
    session = Session.builder.getOrCreate()
    session.file.get(f"@BUILD_STAGE/template/dist/january_ml-0.0.1-py3-none-any.whl", "./dist")