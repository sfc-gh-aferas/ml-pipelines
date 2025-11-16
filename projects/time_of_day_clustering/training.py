from january_ml.utils import load_config

from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline  

from snowflake.snowpark.session import Session

import utils 

config = load_config("time_of_day_clustering")


def main(session: Session) -> dict: 
    df = utils.construct_features(session)

    # Put scaling and clustering in pipeline  
    pipeline = Pipeline(
        steps=[
            ('scaler', StandardScaler()), 
            ('kmeans', KMeans(n_clusters=8, random_state=42, init="k-means++"))
        ])
    df['cluster'] = pipeline.fit_predict(df)

    return {"clusters": df['cluster'].value_counts()}


if __name__ == "__main__":
    session = Session.builder.getOrCreate()

    __return__ = main(session=session)

# The below is all done in a Snowflake notebook right now. We will migrate this to Git as well.
## Register model 
### Register full pipeline as model

## Store cluster assignments in Snowflake
### Debtor UUID, cluster assignment, time of assignment, clustering version, type (initial)

## Store clustering logs in Snowflake 
### Time of creation, number of clusters, cluster centroids, count per cluster, version