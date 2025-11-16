from january_ml.utils import load_config
import utils

from snowflake.snowpark.session import Session

config = load_config("time_of_day_clustering")


def main(session: Session) -> dict:
    df = utils.construct_features(session)
    kmeans = None # TODO: Load in model from Snowflake

    df['cluster'] = kmeans.predict(df)
    
    # TODO: store cluster assignments and metadata in Snowflake
    ## Includes Debtor UUID, cluster, time of assignment, clustering version, type (assignment), distance to nearest centroid
    return {"newly assigned clusters": df['cluster'].value_counts()}


if __name__ == "__main__":
    session = Session.builder.getOrCreate()
    __return__ = main(session=session)