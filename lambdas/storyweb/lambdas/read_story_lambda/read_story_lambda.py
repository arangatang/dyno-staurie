from typing import Dict, List
import boto3
from boto3.dynamodb.types import TypeDeserializer
from storyweb.utils.constants import DDB_TABLE_NAME, error_page
from storyweb.style import load_css
from storyweb.utils.chapters import Chapter
from storyweb.utils.loaders import get_jinja_environment

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
        "body": generate_body(story_id, image, text, choices, chapter),
    }


def from_dynamodb_to_json(item):
    d = TypeDeserializer()
    return {k: d.deserialize(value=v) for k, v in item.items()}


def generate_body(
    story_name: str, image: str, text: str, options: Dict[str, str], chapter: Chapter
):
    environment = get_jinja_environment()
    template = environment.get_template("read_story_template.jinja")
    options = [{"id": key, "text": value} for key, value in options.items()]
    body = template.render(
        text=text,
        options=options,
        story_name=story_name,
        image=image,
        chapter=chapter.chapter_id,
    )
    return body
