from typing import Dict, List
import boto3
from boto3.dynamodb.types import TypeDeserializer
from storyweb.utils.constants import DDB_TABLE_NAME, error_page
from storyweb.style import load_css
from storyweb.utils.chapters import Chapter

ddbclient = boto3.client("dynamodb")


def lambda_handler(event, context):
    load_css("read_story_lambda")
    chapter_id = event.get("queryStringParameters", {}).get("chapter", None)
    story_id = event.get("queryStringParameters", {}).get("story", None)

    if not chapter_id or not story_id:
        # return error_page()
        chapter_id = "1"
        story_id = "test_story"

    print(f"Querying for story: {story_id} and chapter: {chapter_id}")
    response = ddbclient.query(
        TableName=DDB_TABLE_NAME,
        KeyConditionExpression="story = :story AND chapter = :chapter",
        ExpressionAttributeValues={
            ":story": {"S": story_id},
            ":chapter": {"S": chapter_id},
        },
    )

    print("Got response")
    print(response)

    items = response.get("Items", [])
    if len(items) != 1:
        return error_page()

    _chapter = from_dynamodb_to_json(items[0])
    chapter = Chapter(chapter=_chapter)
    image = chapter.image
    text = chapter.text
    choices = chapter.get_choices()

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html"},
        "body": generate_body(story_id, image, text, choices),
    }


def from_dynamodb_to_json(item):
    d = TypeDeserializer()
    return {k: d.deserialize(value=v) for k, v in item.items()}


def generate_body(story_name: str, image: str, text: str, options: Dict[str, str]):
    buttons = "".join(
        [
            f'<div class=center><a id=option_{option_id} class="button" href=https://bne1jvubt0.execute-api.eu-central-1.amazonaws.com/default/story-handler?story={story_name}&chapter={option_id}><center>{option_text}</center></a></div>'
            for option_id, option_text in options.items()
        ]
    )
    content = f"""
    <div>
        <img src='{image}' alt='{text}'/>
        <p id=text>{text}</p>
        {buttons}
    </div>
    """

    body = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>page title</title>
        <style>
        {load_css("read_story_lambda")}
        </style>
    </head>
    <body>
        {content}
    </body>
    </html>
    """
    return body
