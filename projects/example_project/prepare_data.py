from snowflake.snowpark.session import Session
from snowflake.ml.dataset import Dataset
from utils import get_data
from january_ml.utils import version_data

def main(session: Session) -> dict:

    # Get Data
    df = get_data(1000, 5)
    sdf = session.create_dataframe(df)

    # Save as dataset
    ds_name = "TRAIN_DATASET"
    ds = Dataset.create(session=session, name=ds_name, exist_ok=True)
    version_name = version_data(sdf)
    ds_version = ds.create_version(version=version_name, input_dataframe=sdf, label_cols=["y"])

    return {"ds_version": version_name}