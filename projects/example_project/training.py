from sklearn.linear_model import LinearRegression

from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry
from snowflake.ml.dataset import Dataset

from argparse import ArgumentParser

def main(session: Session, version: str) -> dict:

    ds_name = "TRAIN_DATASET"
    ds = Dataset.create(session=session, name=ds_name, exist_ok=True)
    sdf = ds.select_version(version).read.to_snowpark_dataframe()
    df = sdf.to_pandas()

    # Train Model

    X = df.drop(columns=["y"])
    y = df[["y"]]

    lr = LinearRegression()
    lr.fit(X, y)

    # Register model
    reg = Registry(session=session)

    model_name = "TEST_MODEL"
    mv = reg.log_model(
        model=lr, 
        model_name=model_name, 
        sample_input_data=X, 
        target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"])
    
    return {"model_version": mv.version_name}

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--ds_version")
    args = parser.parse_args()

    session = Session.builder.getOrCreate()

    __return__ = main(session=session, version=args.ds_version)