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


    fv = fs.get_feature_view("EXAMPLE_FEATURES",version="1")
    train = fs.read_feature_view(fv).sample(n=1000).select("HOUR","OPENED")
    test = fs.read_feature_view(fv).sample(n=100).select("HOUR","OPENED")

    # Save as datasets
    ds_name = "TRAIN_DATASET"
    ds = Dataset.create(session=session, name=ds_name, exist_ok=True)
    train_version_name = version_data(train)
    ds_version = ds.create_version(version=train_version_name, input_dataframe=train, label_cols=["OPENED"])

    ds_name = "TEST_DATASET"
    ds = Dataset.create(session=session, name=ds_name, exist_ok=True)
    test_version_name = version_data(test)
    ds_version = ds.create_version(version=test_version_name, input_dataframe=test, label_cols=["OPENED"])
    return {"train_version": train_version_name, "test_version": test_version_name}