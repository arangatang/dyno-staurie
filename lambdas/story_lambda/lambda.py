import json
from typing import Dict, Iterable, List
import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer

DDB_TABLE = "arn:aws:dynamodb:eu-central-1:679585051464:table/stories"
DDB_TABLE_NAME = "stories"
LAST_CHAPTER_PREFIX = "FINAL_CHAPTER"
DEFAULT_IMAGE = "https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/1-0.png"

ddbclient = boto3.client("dynamodb")


def lambda_handler(event, context):
    chapter_id = event.get("queryStringParameters", {}).get("chapter", None)
    story_id = event.get("queryStringParameters", {}).get("story", None)

    if not chapter_id or not story_id:
        #return error_page()
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


def error_page():
    return {"statusCode": 404, "body": ""}


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
        {get_style()}
        </style>
    </head>
    <body>
        {content}
    </body>
    </html>
    """
    return body

class ChapterOptions:
    options: List[dict]

    def __init__(self, options: List[dict]):
        self.options = options

    def get_choices(self) -> Dict[str, str]:
        return {
            option.get("next"): option.get("text", "Continue")
            for option in self.options
        }

    def get_path_for_choice(self, option: int) -> str:
        return str(self.options[option].get("next"))
    
    def has_options(self) -> bool:
        return len(self.options) > 0


class Chapter:
    text: str = ""
    options: ChapterOptions = ""
    is_final_chapter: bool = True
    is_work_in_progress: bool = True
    is_start_node: bool = True
    chapter_id = ""
    image = ""
    chapter = {}
    
    
    def __init__(self, chapter: dict):
        # Chapter numbers start with 1.yml
        # For multiple choices subsequent
        self.chapter = chapter
        self.text = chapter.get("text", None)
            
        self.chapter_id = chapter["chapter"]
        self.image = chapter.get("image", DEFAULT_IMAGE)
        if not self.image:
            self.image = DEFAULT_IMAGE
            
        self.options = ChapterOptions(chapter.get("options", []))

        self.is_final_chapter = not self.options.has_options()
        self.is_work_in_progress = chapter.get("is_work_in_progress", False)
        self.is_start_node = chapter.get("is_start_node", False)
        self.disabled_options = chapter.get("disabled_options", [])

    def get_choices(self) -> Dict[str, str]:
        if self.is_final_chapter:
            return {}
        return {k:v for k, v in self.options.get_choices().items() if k not in self.disabled_options}

    def get_path_for_choice(self, choice):
        if self.is_final_chapter:
            raise ValueError(f"No options available for chapter {self.chapter_id}")
        return self.options.get_path_for_choice(choice)
    
    def is_final(self):
        return self.is_final_chapter
    
    def is_start(self):
        return self.is_start_node
    
    def is_unfinished(self):
        return self.is_work_in_progress



def get_style():
    return """
    body {
        background-color: white;
    }

    img {
        display: block;
        width: 500px;
        margin-left: auto;
        margin-right: auto;
        margin-top: 60px;
    }

    p {
        text-align: center;
        line-height: 100px;
        font-size: xx-large;
    }

    .center{
        width: 45%;
        margin: 0 auto;
        background-color: whitesmoke;
        border-style: ridge;
        border-radius: 5px
    }
    
    a:link { text-decoration: none; }
    a:visited { text-decoration: none; }

    .center:hover {
        background-color: floralwhite;
    }
    a {
        font-size: x-large;
        line-height: 50px;
    }
    """