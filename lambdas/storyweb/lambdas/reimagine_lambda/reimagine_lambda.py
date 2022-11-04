import boto3
from storyweb.utils.constants import error_page
import hashlib
import json
import uuid

sqs = boto3.resource("sqs")
queue = sqs.get_queue_by_name(QueueName="stories-texts-to-process.fifo")


def lambda_handler(event, context):

    try:
        print(event)
        params = event.get("queryStringParameters", {})
        userId = params.get("userId")
        text = params.get("text")
    except Exception:
        return error_page("Invalid get parameters")
    text_hash = hashlib.md5(bytes(text, encoding="utf-8")).hexdigest()

    imageUrl = f"https://story-images.s3.eu-central-1.amazonaws.com/reimagine/{userId}/images/{text_hash}-0.png"

    try:
        print("Sending message to SQS")
        dedupestring = f"{userId}#{text_hash}"
        queue.send_message(
            MessageBody=text,
            MessageAttributes={
                "userId": {"StringValue": userId, "DataType": "String"},
                "imageUrl": {"StringValue": imageUrl, "DataType": "String"},
                "imageId": {"StringValue": text_hash, "DataType": "String"},
            },
            MessageGroupId=dedupestring,
            MessageDeduplicationId=uuid.uuid4().hex,
        )

    except Exception as e:
        print(
            f"Unable to send message to queue: {text} userId: {userId}, exception was: {e}"
        )
    return generate_response(imageUrl)


def generate_response(imageUrl):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"imageUrl": imageUrl}),
    }
