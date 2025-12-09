import yaml
import os
import json
import hashlib
from snowflake.ml.feature_store import FeatureView, FeatureStore
import snowflake.snowpark.functions as F
from snowflake.snowpark import DataFrame

def version_data(df:DataFrame) -> str:

    """Computes an md5 hash on the data itself to be used as a dataset version."""

    return hashlib.md5(str(df.select_expr("HASH_AGG(*)")).encode('utf-8')).hexdigest().upper()