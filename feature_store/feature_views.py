from snowflake.snowpark import Session, DataFrame
import snowflake.snowpark.functions as F

def create_example_features(session: Session) -> DataFrame:


    df = session.table('DEBTSY.AUTOMAT_REPORTING.REP_OUTBOUND_COMMS')
    df = df.with_column("HOUR", F.date_part("HOUR","CREATED_AT"))
    df = df.with_column("OPENED", F.col("OPEN_AT").is_not_null())
    df = df.select("BORROWERCOMM_UUID","CREATED_AT","HOUR","OPENED")
    
    return df