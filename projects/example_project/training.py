from sklearn.linear_model import LinearRegression

from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry
from snowflake.ml.dataset import Dataset

from argparse import ArgumentParser

def main(session: Session, train_version: str, test_version: str) -> dict:

    ds_name = "TRAIN_DATASET"
    ds = Dataset.create(session=session, name=ds_name, exist_ok=True)
    sdf = ds.select_version(train_version).read.to_snowpark_dataframe()
    df = sdf.to_pandas()

    # Train Model

    X_train = df.drop(columns=["OPENED"])
    y_train = df[["OPENED"]]

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    ds_name = "TEST_DATASET"
    ds = Dataset.create(session=session, name=ds_name, exist_ok=True)
    sdf = ds.select_version(test_version).read.to_snowpark_dataframe()
    df = sdf.to_pandas()

    X_test = df.drop(columns=["OPENED"])
    y_test = df[["OPENED"]]

    score = lr.score(X_test, y_test)

    # Register model
    reg = Registry(session=session)

    model_name = "TEST_MODEL"
    mv = reg.log_model(
        model=lr, 
        model_name=model_name, 
        sample_input_data=X_train, 
        target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        metrics={"score": score}
    )
    
    return {"model_version": mv.version_name}

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--train_version")
    parser.add_argument("--test_version")
    args = parser.parse_args()

    session = Session.builder.getOrCreate()

    __return__ = main(session=session, train_version=args.train_version, test_version=args.test_version)