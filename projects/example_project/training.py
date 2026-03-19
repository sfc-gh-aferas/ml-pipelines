from sklearn.linear_model import LogisticRegression
from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry
from snowflake.ml.dataset import Dataset
from argparse import ArgumentParser

def main(session: Session, train_version: str, test_version: str) -> dict:

    ds_name = "TRAIN_DATASET"
    ds = Dataset.create(session=session, name=ds_name, exist_ok=True)
    sdf = ds.select_version(train_version).read.to_snowpark_dataframe().fillna(0)
    df = sdf.to_pandas()

    # Train Model

    X_train = df.drop(columns=["MORTGAGERESPONSE"])
    y_train = df[["MORTGAGERESPONSE"]]

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    test_ds_name = "TEST_DATASET"
    test_ds = Dataset.create(session=session, name=test_ds_name, exist_ok=True)
    test_sdf = test_ds.select_version(test_version).read.to_snowpark_dataframe().fillna(0)
    test_df = test_sdf.to_pandas()

    X_test = test_df.drop(columns=["MORTGAGERESPONSE"])
    y_test = test_df[["MORTGAGERESPONSE"]]

    score = lr.score(X_test, y_test)

    # Register model
    reg = Registry(session=session)
    model_name = "MODEL_EX2"
    mv = reg.log_model(
        model=lr, 
        model_name=model_name, 
        sample_input_data=sdf.drop("MORTGAGERESPONSE").limit(100), 
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