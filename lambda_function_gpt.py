
from customer_workflow import app

def lambda_handler(event, context):
    query = event.get("query", "preferences of customer")
    result = app.invoke({"query": query})
    return {
        "statusCode": 200,
        "body": result
    }
