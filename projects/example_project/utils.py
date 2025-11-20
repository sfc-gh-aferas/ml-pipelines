from snowflake.snowpark import Session
import numpy as np
import pandas as pd

def get_stage_packages():
    session = Session.builder.getOrCreate()
    session.file.get("@BUILD_STAGE/example_project/dist/january_ml-0.0.1-py3-none-any.whl", "./dist")