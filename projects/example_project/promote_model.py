from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry


"""
Requires creation of notification integration in Snowflake.
See documentation at 
https://docs.snowflake.com/en/user-guide/notifications/email-notifications#label-create-email-notification-integration
""" 


def main(session: Session, model_version: str) -> None:

    # Get model
    reg = Registry(session=session)

    model_name = "TEST_MODEL"

    base_model = reg.get_model(model_name)
    mv = base_model.version(model_version)

    score = mv.show_metrics()["score"]

    if score < 0.98:
        #session.sql("""
        #    CALL SYSTEM$SEND_SNOWFLAKE_NOTIFICATION(
        #        SNOWFLAKE.NOTIFICATION.TEXT_PLAIN(
        #            'Model quality check failed to meet the threshold.'
        #        ),
        #        SNOWFLAKE.NOTIFICATION.INTEGRATION('MY_EMAIL_INTEGRATION')
        #    );
        #""").collect()
        return "Notification sent"
    else:
        base_model.default = mv
        return "Model quality check passed"