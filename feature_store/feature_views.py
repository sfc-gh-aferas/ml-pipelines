from snowflake.snowpark import Session, DataFrame
import snowflake.snowpark.functions as F
from datetime import datetime

def create_example_features(session: Session) -> DataFrame:

    df = session.table("MORTGAGE_LENDING_DEMO_DATA")

    #Get current date and time
    current_time = datetime.now()
    df_max_time = datetime.strptime(str(df.select(F.max("TS")).collect()[0][0]), "%Y-%m-%d %H:%M:%S.%f")

    #Find delta between latest existing timestamp and today's date
    timedelta = current_time- df_max_time

    #Update timestamps to represent last ~1 year from today's date
    df.select(F.min(F.date_add(F.to_timestamp("TS"), timedelta.days-1)), F.max(F.date_add(F.to_timestamp("TS"), timedelta.days-1)))

    #Create a dict with keys for feature names and values containing transform code

    feature_eng_dict = dict()

    #Timstamp features
    feature_eng_dict["TIMESTAMP"] = F.date_add(F.to_timestamp("TS"), timedelta.days-1)
    feature_eng_dict["MONTH"] = F.month("TIMESTAMP")
    feature_eng_dict["DAY_OF_YEAR"] = F.dayofyear("TIMESTAMP") 
    feature_eng_dict["DOTW"] = F.dayofweek("TIMESTAMP")

    # df= df.with_columns(feature_eng_dict.keys(), feature_eng_dict.values())

    #Income and loan features
    feature_eng_dict["LOAN_AMOUNT"] = F.col("LOAN_AMOUNT_000s")*1000
    feature_eng_dict["INCOME"] = F.col("APPLICANT_INCOME_000s")*1000
    feature_eng_dict["INCOME_LOAN_RATIO"] = F.col("INCOME")/F.col("LOAN_AMOUNT")

    df = df.with_columns(feature_eng_dict.keys(), feature_eng_dict.values())
    feature_df = df.select(["LOAN_ID"]+list(feature_eng_dict.keys()))

    return feature_df