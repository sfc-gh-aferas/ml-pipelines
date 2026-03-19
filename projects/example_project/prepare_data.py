from snowflake.snowpark.session import Session
from snowflake.ml.dataset import Dataset
from snowflake.ml.feature_store import FeatureStore, CreationMode
from ml_utils.utils import version_data

def main(session: Session) -> dict:
    
    # Get Data
    fs = FeatureStore(
        session=session,
        database=session.get_current_database(),
        name=session.get_current_schema(),
        default_warehouse=session.get_current_warehouse(),
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST
    )


    fv = fs.get_feature_view("EXAMPLE_FEATURES",version="3")
    train = fs.read_feature_view(fv).sample(n=1000).select("INCOME","MORTGAGERESPONSE")
    test = fs.read_feature_view(fv).sample(n=100).select("INCOME","MORTGAGERESPONSE")

    # Save as datasets
    ds_name = "TRAIN_DATASET"
    ds = Dataset.create(session=session, name=ds_name, exist_ok=True)
    train_version_name = version_data(train)
    ds_version = ds.create_version(version=train_version_name, input_dataframe=train, label_cols=["MORTGAGERESPONSE"])

    ds_name = "TEST_DATASET"
    ds = Dataset.create(session=session, name=ds_name, exist_ok=True)
    test_version_name = version_data(test)
    ds_version = ds.create_version(version=test_version_name, input_dataframe=test, label_cols=["MORTGAGERESPONSE"])
    return {"train_version": train_version_name, "test_version": test_version_name}