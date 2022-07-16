from faulthandler import disable
from typing import Dict, List
import boto3
from boto3.dynamodb.types import TypeDeserializer


DDB_TABLE = "arn:aws:dynamodb:eu-central-1:679585051464:table/stories"
DDB_TABLE_NAME = "stories"


ddbclient = boto3.client("dynamodb")
table = boto3.resource('dynamodb').Table(DDB_TABLE_NAME)
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
    
    def __init__(self, chapter: dict, filter_choices: bool = True):
        # Chapter numbers start with 1.yml
        # For multiple choices subsequent
        self.chapter = chapter
        self.text = chapter.get("text", None)
        self.filter_choices = filter_choices

        self.chapter_id = chapter["chapter"]
        self.image = chapter.get("image", "")
        self.options = ChapterOptions(chapter.get("options", []))

        self.is_final_chapter = not self.options.has_options()
        self.is_work_in_progress = chapter.get("is_work_in_progress", False)
        self.is_start_node = chapter.get("is_start_node", False)
        self.disabled_options = chapter.get("disabled_options", [])

    def get_choices(self) -> Dict[str, str]:
        if self.is_final_chapter:
            return {}
        if self.filter_choices:
            return {k:v for k, v in self.options.get_choices().items() if k not in self.disabled_options}
        else:
            return {k:v for k, v in self.options.get_choices().items()}

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


def lambda_handler(event, context):
    params = event.get("queryStringParameters", {})
    if not(params):
        return error_page("Invalid get parameters")

    chapter_id = params.get("chapter_id", None)
    chapter_text = params.get("chapter_text", None)
    story_id = params.get("story_id", None)
    
    if not chapter_text or not chapter_id or not story_id:
        return error_page(reason="text, chapter id and story id are required")
    
    new_option_id = params.get("new_option_id", None)
    new_option_text = params.get("new_option_text", None)
    
    existing_options = {
        k.replace("option_", ""): v
        for k, v in params.items()
        if k.startswith("option_")
    }

    for i in (chapter_id, story_id, chapter_text, new_option_id, new_option_text, *existing_options.keys(), *existing_options.values()):
        check_input(i)

    chapter = get_chapter(story_id=story_id, chapter_id=chapter_id)
    disabled_options = chapter.disabled_options
    
    if len(chapter.get_choices()) != len(existing_options):
        return error_page(reason="invalid amount of options")
    
    print("Existing options: ", existing_options)
    for key in chapter.get_choices().keys():
        # protect against URL tampering, rn we only support adding options or editing existing ones
        # otherwise the graph could inadvertently be split into two causing the end of the world
        if str(key) not in existing_options.keys():
            return error_page(reason=f"passed key {key} does not match the data stored for the chapter: {existing_options.keys()}")

    
    # Create a new chapter if new_option_text is set and no existing chapter with that id exists
    if new_option_text and new_option_id:
        if not isinstance(get_chapter(story_id=story_id, chapter_id=new_option_id), Chapter):
            # Upload new chapter before updating existing chapter to avoid dead nodes
            table.put_item(
                Item=dict(
                    story=story_id,
                    chapter=new_option_id,
                    is_work_in_progress=True
                )
            )
            disabled_options.append(new_option_id)
        existing_options[new_option_id] = new_option_text
            
    

    # Create the updated chapter
    ddb_item = dict(
        story = story_id,
        chapter = chapter_id,
        image = chapter.image,
        text = chapter.text if not chapter_text else chapter_text,
    )
    
    if existing_options:
        ddb_item["options"] = [{"next":key, "text": value} for key, value in existing_options.items()]
    if chapter.is_start():
        ddb_item["is_start_node"] = True
    if disabled_options:
        ddb_item["disabled_options"] = disabled_options

    # Upload the updated chapter
    print(f"Uploading chapter: {ddb_item}")
    table.put_item(
        Item=ddb_item
    )
    
    # Return HTML with 2 buttons, one to view the new chapter and one to return to the edit screen for the newly created chapter

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html"},
        "body": generate_body(
            story_id=story_id,
            chapter_id=chapter_id,
            new_option_id=new_option_id if new_option_text else None,
        ),
    }

def get_chapter(story_id, chapter_id):
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
    return Chapter(chapter=_chapter, filter_choices=False)

def check_input(text):
    # TODO what would be deemed malicious / could we check this before sending request?
    pass

def error_page(reason: str = ''):
    return {"statusCode": 404, "body": f"{reason}"}


def from_dynamodb_to_json(item):
    d = TypeDeserializer()
    return {k: d.deserialize(value=v) for k, v in item.items()}


def generate_body(story_id, chapter_id, new_option_id):
    html = get_html()
    style = get_style()
    
    new_chapter_button = f'<div class=center><a id="edit_new_chapter_button class="button" href=https://bne1jvubt0.execute-api.eu-central-1.amazonaws.com/default/edit?story={story_id}&chapter={new_option_id}><center>Edit the newly created chapter</center></a></div>' if new_option_id else ''
    view_updated_chapter_button = f'<div class=center><a id="view_edited_chapter_button class="button" href=https://bne1jvubt0.execute-api.eu-central-1.amazonaws.com/default/story-handler?story={story_id}&chapter={chapter_id}><center>View your changes to chapter {chapter_id}</center></a></div>'
    buttons = [new_chapter_button, view_updated_chapter_button]
    formatted = html % (style, "".join(buttons))
    # print(formatted)
    return formatted



def get_style():
    return """
   body {
        background-color: white;
        margin: 0;
    }

    img {
        display: block;
        margin-left: 11vh;
        padding-right: 11vh;
        width: 30vh;
        float: left;
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

    .edit-window {
        display: none;
        height: 50%;
    }
    
    html {
        height: 100%;
    }
    
    body {
        min-height: 100%;
    }
    
    #mynetwork {
        width: 100%;
        height: 100%;
        border: 1px solid lightgray;
        min-height: 100%;
    }
    
    center { 
        width: 100vw;
        height: 50vh;
    }
    
    .edit-window {
        padding-top: 10vh;
    }
    
    .edit-window > form {
        padding-top: 2vh;
        padding-right: 2vh;
        display: flex;
        flex-direction: column;
    }
    """


def get_html():
    return """
    <!DOCTYPE html>
    <html lang="en">

    <head>
        <title>Network</title>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style type="text/css">
        %s
        </style>
    </head>

    <body>
        <div>
            %s
        <div>
    </body>

    </html>
    """

if __name__ == "__main__":
    lambda_handler(
        {
            "version": "1.0",
            "resource": "/update_story",
            "path": "/default/update_story",
            "httpMethod": "GET",
            "headers": {
                "Content-Length": "0",
                "Host": "bne1jvubt0.execute-api.eu-central-1.amazonaws.com",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
                "X-Amzn-Trace-Id": "Root=1-62cdf9ca-6555c8a93bb1c77d34dbff10",
                "X-Forwarded-For": "109.40.242.10",
                "X-Forwarded-Port": "443",
                "X-Forwarded-Proto": "https",
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
                "accept-encoding": "gzip, deflate, br",
                "accept-language": "en-US,en;q=0.9,sv-SE;q=0.8,sv;q=0.7",
                "dnt": "1",
                "sec-ch-ua": '".Not/A)Brand";v="99", "Google Chrome";v="103", "Chromium";v="103"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "document",
                "sec-fetch-mode": "navigate",
                "sec-fetch-site": "none",
                "sec-fetch-user": "?1",
                "upgrade-insecure-requests": "1",
            },
            "multiValueHeaders": {
                "Content-Length": ["0"],
                "Host": ["bne1jvubt0.execute-api.eu-central-1.amazonaws.com"],
                "User-Agent": [
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"
                ],
                "X-Amzn-Trace-Id": ["Root=1-62cdf9ca-6555c8a93bb1c77d34dbff10"],
                "X-Forwarded-For": ["109.40.242.10"],
                "X-Forwarded-Port": ["443"],
                "X-Forwarded-Proto": ["https"],
                "accept": [
                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"
                ],
                "accept-encoding": ["gzip, deflate, br"],
                "accept-language": ["en-US,en;q=0.9,sv-SE;q=0.8,sv;q=0.7"],
                "dnt": ["1"],
                "sec-ch-ua": [
                    '".Not/A)Brand";v="99", "Google Chrome";v="103", "Chromium";v="103"'
                ],
                "sec-ch-ua-mobile": ["?0"],
                "sec-ch-ua-platform": ['"Windows"'],
                "sec-fetch-dest": ["document"],
                "sec-fetch-mode": ["navigate"],
                "sec-fetch-site": ["none"],
                "sec-fetch-user": ["?1"],
                "upgrade-insecure-requests": ["1"],
            },
            "queryStringParameters": {
                "_text": "One night, the boy woke up from a loud cracking sound just outside his house.",
                "chapter_id": "3",
                "new_option_id": "4",
                "new_option_text": "",
                "option_3.1": "The boy went outside to investigate",
                "option_3.2": "The boy, who was used to these kind of sounds, quickly fell back asleep",
                "story_id": "test_story",
            },
            "multiValueQueryStringParameters": {
                "_text": [
                    "One night, the boy woke up from a loud cracking sound just outside his house."
                ],
                "chapter_id": ["3"],
                "new_option_id": ["4"],
                "new_option_text": [""],
                "option_3.1": ["The boy went outside to investigate"],
                "option_3.2": [
                    "The boy, who was used to these kind of sounds, quickly fell back asleep"
                ],
                "story_id": ["test_story"],
            },
            "requestContext": {
                "accountId": "679585051464",
                "apiId": "bne1jvubt0",
                "domainName": "bne1jvubt0.execute-api.eu-central-1.amazonaws.com",
                "domainPrefix": "bne1jvubt0",
                "extendedRequestId": "VLP3sgx6FiAEMHw=",
                "httpMethod": "GET",
                "identity": {
                    "accessKey": None,
                    "accountId": None,
                    "caller": None,
                    "cognitoAmr": None,
                    "cognitoAuthenticationProvider": None,
                    "cognitoAuthenticationType": None,
                    "cognitoIdentityId": None,
                    "cognitoIdentityPoolId": None,
                    "principalOrgId": None,
                    "sourceIp": "109.40.242.10",
                    "user": None,
                    "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
                    "userArn": None,
                },
                "path": "/default/update_story",
                "protocol": "HTTP/1.1",
                "requestId": "VLP3sgx6FiAEMHw=",
                "requestTime": "12/Jul/2022:22:46:34 +0000",
                "requestTimeEpoch": 1657665994634,
                "resourceId": "ANY /update_story",
                "resourcePath": "/update_story",
                "stage": "default",
            },
            "pathParameters": None,
            "stageVariables": None,
            "body": None,
            "isBase64Encoded": False,
        },
        None,
    )
