from snowflake.snowpark import Session
import numpy as np
import pandas as pd

def get_stage_packages():
    session = Session.builder.getOrCreate()
    session.file.get("@PACKAGE_STAGE/dist/january_ml-0.0.1-py3-none-any.whl", "./dist")

def get_data(n_samples, n_features):
    rand = np.random.default_rng(42)
    data = rand.random(size=(n_samples, n_features+1))
    df = pd.DataFrame(data, columns=["input_feature_"+str(i) for i in range(n_features)]+["y"])
    return df