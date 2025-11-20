from snowflake.snowpark.session import Session
from snowflake.ml.dataset import Dataset
from snowflake.ml.feature_store import FeatureStore, CreationMode
from january_ml.utils import version_data
from january_ml.constants import FEATURE_SCHEMA

def main(session: Session) -> dict:

    # Get Data

    fs = FeatureStore(
        session=session,
        database=session.get_current_database(),
        name=FEATURE_SCHEMA,
        default_warehouse=session.get_current_warehouse(),
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST
    )

    fv = fs.get_feature_view("EXAMPLE_FEATURES",version="590DC9DFB32D518D5AD2CA1CC5537D6D")
    df = fs.read_feature_view(fv).sample(n=1000)

    # Save as dataset
    ds_name = "TRAIN_DATASET"
    df = df.select("HOUR","OPENED")
    ds = Dataset.create(session=session, name=ds_name, exist_ok=True)
    version_name = version_data(df)
    ds_version = ds.create_version(version=version_name, input_dataframe=df, label_cols=["OPENED"])

    return {"ds_version": version_name}