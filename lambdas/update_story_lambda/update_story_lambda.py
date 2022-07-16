from typing import Dict, List
import boto3
from boto3.dynamodb.types import TypeDeserializer

ddbclient = boto3.client("dynamodb")


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


class Chapter:
    text: str = ""
    options: ChapterOptions = ""
    is_final_chapter: bool = True
    chapter_id = ""
    image = ""

    def __init__(self, chapter: dict):
        # Chapter numbers start with 1.yml
        # For multiple choices subsequent
        if not chapter.get("text"):
            raise ValueError("Found chapter without text", chapter)
        else:
            self.text = chapter["text"]

        self.chapter_id = chapter["chapter"]
        self.image = chapter["image"]
        self.options = ChapterOptions(chapter.get("options", []))
        self.is_final_chapter = (
            self.chapter_id.startswith(LAST_CHAPTER_PREFIX) or not self.options
        )

    def get_choices(self) -> Dict[str, str]:
        if self.is_final_chapter:
            return {}
        return self.options.get_choices()

    def get_path_for_choice(self, choice):
        if self.is_final_chapter:
            raise ValueError(f"No options available for chapter {self.chapter_id}")
        return self.options.get_path_for_choice(choice)


def lambda_handler(event, context):
    params = event.get("queryStringParameters", {})
    chapter_id = params.get("chapter_id", None)
    new_option_id = params.get("new_option_id", None)
    new_option_text = params.get("new_option_text", None)
    story_id = params.get("story_id", None)
    options = {
        k.replace("option_", ""): v
        for k, v in params.items()
        if k.startswith("option_")
    }

    # TODO
    # 1. parse the passed data for obvious maliciousness
    # 2. for each field, update the table if different
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html"},
        "body": generate_body(
            get_edit_windows(chapters, story_id=story_id, current_chapter=chapter_id),
            nodes,
            edges,
            chapter_id=chapter_id,
        ),
    }


def error_page():
    return {"statusCode": 404, "body": ""}


def from_dynamodb_to_json(item):
    d = TypeDeserializer()
    return {k: d.deserialize(value=v) for k, v in item.items()}


def get_edge(_from, to):
    return '{ from: %s, to: %s, arrows: { to: { enabled: true, type: "arrow" }}},' % (
        _from,
        to,
    )


def get_node(id: str, label: str):
    green = "#00FF00"
    red = "#FF0000"
    color = ""
    node_color = "#4DA4EA"
    if id.startswith("-"):
        node_color = red
    elif id.startswith("1"):
        node_color = green

    if node_color:
        color = 'color: { background: "%s"}' % node_color
    return '{ id: "%s", label: "%s", %s },' % (id, label, color)


def get_nodes(chapters: Dict[str, Chapter]):
    return "\n".join([get_node(key, key) for key in chapters.keys()])


def get_edit_window_existing_option(option_id: str, option_text: str):
    return """
        <label for="{option_id}">Choice leading to chapter: {option_id}</label>
        <input type="text" id="option_{option_id}" name="option_{option_id}" default="{option_text}" value="{option_text}"><br><br>
    """.format(
        option_id=option_id, option_text=option_text
    )


def get_edit_window(
    chapter: Chapter, next_choice_id: int, story_id: str, should_show: bool
):
    options = []
    for option_id, option_text in chapter.get_choices().items():
        options.append(get_edit_window_existing_option(option_id, option_text))

    return get_edit_window_html().format(
        CHAPTER_ID=chapter.chapter_id,
        IMAGE=chapter.image,
        EXISTING_OPTIONS="""
        """.join(
            options
        ),
        STORY_ID=story_id,
        NEXT_CHOICE_ID=next_choice_id,
        CHAPTER_TEXT=chapter.text,
        DISPLAY=' edit-window-clicked" style="display: block' if should_show else "",
    )


def get_edit_windows(chapters: Dict[str, Chapter], story_id: str, current_chapter: str):
    next_choice_id = max([int(id.split(".")[0]) for id in chapters]) + 1
    return "".join(
        [
            get_edit_window(
                chapter,
                next_choice_id,
                story_id=story_id,
                should_show=current_chapter == chapter_id,
            )
            for chapter_id, chapter in chapters.items()
        ]
    )


def get_edges(chapters: Dict[str, Chapter]):
    edges = []
    for chapter_id, chapter in chapters.items():
        for choice in chapter.get_choices().keys():
            edges.append(get_edge(chapter_id, choice))
    return """
    """.join(
        edges
    )


def select_node(node_id):
    return node_id if node_id else ""


def generate_body(edit_windows, nodes, edges, chapter_id):
    html = get_html()
    style = get_style()
    formatted = html % (style, edit_windows, nodes, edges, select_node(chapter_id))
    # print(formatted)
    return formatted


DDB_TABLE = "arn:aws:dynamodb:eu-central-1:679585051464:table/stories"

DDB_TABLE_NAME = "stories"

LAST_CHAPTER_PREFIX = "-"


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
        <center>
            <div id="mynetwork"></div>
        </center>
        <div id="edit_windows">
            %s
        </div>
        <script type="text/javascript">
            // create an array with nodes
            var nodes = new vis.DataSet([
                %s
            ]);

            // create an array with edges
            var edges = new vis.DataSet([
                %s
            ]);

            // create a network
            var container = document.getElementById("mynetwork");
            var data = {
                nodes: nodes,
                edges: edges,
            };
            var options = { layout : { randomSeed: 1994 }};
            var network = new vis.Network(container, data, options);
            network.selectNodes([%s]);

            network.on( 'click', function(properties) {
                var ids = properties.nodes;
                var clickedNodes = nodes.get(ids);
                console.log('clicked nodes:', clickedNodes);
                if(clickedNodes.length == 1){
                    console.log("clicked node:", "edit-window-"+clickedNodes[0].id)
                    var clicked = document.getElementById("edit-window-"+clickedNodes[0].id);
                    clicked.style.display = "block";
                } 
                var clickedClassName = "edit-window-clicked"
                var previouslyClicked = document.getElementsByClassName(clickedClassName);
                if(previouslyClicked.length == 1){
                    previouslyClicked[0].style.display = "none";
                    previouslyClicked[0].classList.remove(clickedClassName);
                }
                clicked.classList.add(clickedClassName);   
            });
        </script>
    </body>

    </html>
    """


def get_edit_window_html():
    return """
    <div class="edit-window{DISPLAY}" id="edit-window-{CHAPTER_ID}">
        <img src="{IMAGE}" loading="lazy">
        <form action="/process_edit_story" method="get">
            <label for="_text">Chapter text: </label>
            <input type="text" id="_text" name="_text" value="{CHAPTER_TEXT}"><br><br>
            {EXISTING_OPTIONS}
            <label for="new_option">New choice text: </label>
            <input type="text" id="new_option_text" name="new_option_text"><br><br>
            
            <input type="hidden" id="new_option_id" name="new_option_id" value="{NEXT_CHOICE_ID}"><br><br>
            <input type="hidden" name="story_id" value={STORY_ID} class="hidden">
            <input type="hidden" name="chapter_id" value={CHAPTER_ID} class="hidden">

            <input type="submit" value="Submit">
        </form>
    </div>
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
