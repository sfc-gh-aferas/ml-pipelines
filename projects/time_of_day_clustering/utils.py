## TODO: turn into feature store, handle missing values elegantly, warning/error logs

import os

from dotenv import load_dotenv
from snowflake.snowpark import Session

load_dotenv()


def get_sql_features(session: Session):
    """
    Retrieve debt-level data from Snowflake.
    """ 
    query = f"""
        with zip_cte as (
        select 
            ad.debtor_uuid,
            zip.median_household_income as zip_median_household_income
        from debtsy.automat.most_recent_address as ad 
        left join debtsy.automat.zipcode_income_and_population_density as zip 
        on (left(ad.address_zip, 5) = lpad(to_varchar(zip.zipcode), 5, '0'))
    ), address_cte as (
        select
            sa.debtor_uuid,
            count(*) as address_count,
            iff(
                    min(sa.reported_at) is null, 
                    min(sa.created_at), 
                    min(sa.reported_at)
                ) as min_date,
            max(sa.created_at) as max_date,
            iff(
                    min_date = max_date, 
                    1, 
                    ceil(datediff(second, min_date, max_date)/60/60/24/365.25)
                ) as time_year,
            iff(
                    time_year < 1, 
                    NULL, 
                    address_count/time_year
                ) as addresses_per_year
        from debtsy.automat.stg_address as sa 
        group by sa.debtor_uuid
    )
    select
        ld.debtor_uuid,
        ld.debt_uuid,
        ld.original_balance,
        ld.payable_balance,
        ld.ord_placement,
        ld.paid_so_far,
        iff(ld.settlement_amount is null, ld.original_balance, ld.settlement_amount) as settlement_amount,
        ld.created_at,
        datediff(day, ld.created_at, getdate()) as days_since_creation,
        ld.last_paid_date,
        datediff(day, ld.last_paid_date, getdate()) as days_since_last_payment,
        ac.addresses_per_year,
        iff(ld.client_last_payment_date is null, 1, 0) as first_paid_default,
        ifnull(datediff(month, ld.charged_off_date, getdate()), datediff(month, ld.placed_at, getdate())) as loan_age_month,
        zc.zip_median_household_income,
        iff(ld.loan_type = 'credit_card', 1, 0) as loan_type_credit_card,
        iff(ld.loan_type = 'bnpl', 1, 0) as loan_type_bnpl
    from debtsy.automat.latest_debt as ld 
    left join debtsy.automat.stg_mab_clusters as mab 
    on (ld.debtor_uuid = mab.debtor_uuid)
    left join debtsy.automat.aggr_pl_attributes as apa 
    on (ld.debt_uuid = apa.debt_uuid)
    left join zip_cte as zc 
    on (ld.debtor_uuid = zc.debtor_uuid)
    left join address_cte as ac 
    on (ld.debtor_uuid = ac.debtor_uuid)
    where ld.status = 'placed' and mab.debtor_uuid is null 
    """ 
    df = session.sql(query).to_pandas()
    return df 


def handle_missing_values(df): 
    """
    Helper function to handle missing values in the debt-level features.
    """
    df['ORD_PLACEMENT'] = df['ORD_PLACEMENT'].fillna(df['ORD_PLACEMENT'].mode()[0])
    df['ADDRESSES_PER_YEAR'] = df['ADDRESSES_PER_YEAR'].fillna(0)
    df['ZIP_MEDIAN_HOUSEHOLD_INCOME'] = df['ZIP_MEDIAN_HOUSEHOLD_INCOME'].fillna(df['ZIP_MEDIAN_HOUSEHOLD_INCOME'].mean())
    df['DAYS_SINCE_LAST_PAYMENT'] = df['DAYS_SINCE_LAST_PAYMENT'].fillna(10000)
    
    return df 


def get_debt_features(df): 
    """
    Create additional debt-level features and handle missing values.
    """
    df['SETTLEMENT_FORGIVENESS_FRACTION'] = (df.ORIGINAL_BALANCE - df.SETTLEMENT_AMOUNT) / df.ORIGINAL_BALANCE
    df['HAS_SETTLEMENT'] = False 
    df.loc[df['SETTLEMENT_FORGIVENESS_FRACTION'] > 0, 'HAS_SETTLEMENT'] = True 

    df['CREDIT_CARD_DEBT'] = df['LOAN_TYPE_CREDIT_CARD'] * df['ORIGINAL_BALANCE']
    df['BNPL_DEBT'] = df['LOAN_TYPE_BNPL'] * df['ORIGINAL_BALANCE']

    df['PAID_FRACTION'] = df['PAID_SO_FAR'] / df['PAYABLE_BALANCE']
    df = df[df['PAID_FRACTION'] < 2] # remove outliers

    return handle_missing_values(df)


def get_debtor_features(df): 
    """
    Aggregate debt-level features to the debtor level.
    """
    df_debtor = df.groupby("DEBTOR_UUID").agg(
        total_debt = ("ORIGINAL_BALANCE", "sum"),
        total_paid = ("PAID_SO_FAR", "sum"),

        avg_ord_placement = ("ORD_PLACEMENT", "mean"),
        max_ord_placement = ("ORD_PLACEMENT", "max"),

        days_since_oldest_debt = ("DAYS_SINCE_CREATION", "max"),
        days_since_newest_debt = ("DAYS_SINCE_CREATION", "min"),
        days_since_last_payment = ("DAYS_SINCE_LAST_PAYMENT", "min"),

        addresses_per_year = ("ADDRESSES_PER_YEAR", "mean"),
        zip_median_household_income = ("ZIP_MEDIAN_HOUSEHOLD_INCOME", "mean"),

        avg_first_paid_default = ("FIRST_PAID_DEFAULT", "mean"),
        avg_loan_age_months = ("LOAN_AGE_MONTH", "mean"),

        settlement_rate = ("HAS_SETTLEMENT", "mean"),
        nr_debts = ("DEBT_UUID", "count"),

        credit_card_debt=("CREDIT_CARD_DEBT", "sum"),
        bnpl_debt=("BNPL_DEBT", "sum")
    )

    df_debtor['paid_fraction'] = df_debtor['total_paid'] / df_debtor['total_debt']
    df_debtor['credit_card_fraction'] = df_debtor['credit_card_debt'] / df_debtor['total_debt']
    df_debtor['bnpl_fraction'] = df_debtor['bnpl_debt'] / df_debtor['total_debt']

    return df_debtor 


def get_cluster_features(df): 
    """
    Retrieve cluster-level features for each debtor.
    """
    cluster_features = [
        "total_debt",
        "total_paid",
        "avg_ord_placement",
        "days_since_oldest_debt",
        "days_since_newest_debt",
        "days_since_last_payment",
        "addresses_per_year",
        "zip_median_household_income",
        "avg_first_paid_default",
        "avg_loan_age_months",
        "settlement_rate",
        "nr_debts",
        "credit_card_fraction",
        "bnpl_fraction",
        "paid_fraction"       
    ]

    return df[cluster_features]
    

def construct_features(session: Session):
    """
    Retrieve dataframe with features for clustering and cluster assignments.
    """
    df = get_sql_features(session)
    df = get_debt_features(df)
    df_debtor = get_debtor_features(df)
    df_cluster = get_cluster_features(df_debtor)
    
    return df_cluster