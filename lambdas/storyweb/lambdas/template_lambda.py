from typing import Dict, List
import boto3
from boto3.dynamodb.types import TypeDeserializer
from storyweb.utils.constants import DDB_TABLE_NAME, error_page
from storyweb.style import load_css
from storyweb.utils.chapters import Chapter
from storyweb.utils.loaders import get_jinja_environment

ddbclient = boto3.client("dynamodb")


def lambda_handler(event, context):
    # Extract some query string params if needed
    # chapter_id = event.get("queryStringParameters", {}).get("chapter", None)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html"},
        "body": generate_body(),
    }


def generate_body():
    environment = get_jinja_environment()
    # replace the template with whatever page you are building
    template = environment.get_template("base_website.jinja")
    return template.render()
