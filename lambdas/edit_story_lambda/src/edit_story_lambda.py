from typing import Dict, List
import boto3
from boto3.dynamodb.types import TypeDeserializer

#ddbclient = boto3.client("dynamodb")

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
        self.is_final_chapter = self.chapter_id.startswith(LAST_CHAPTER_PREFIX)
        self.image = chapter["image"]
        
        if not self.is_final_chapter:
            # if no option given, then always set next chapter as current chapter + 1
            self.options = ChapterOptions(chapter.get("options"))

    def get_choices(self) -> Dict[str, str]:
        if self.is_final_chapter:
            return {}

        return self.options.get_choices()

    def get_path_for_choice(self, choice):
        return self.options.get_path_for_choice(choice)



def lambda_handler(event, context):
    #chapter_id = event.get("queryStringParameters", {}).get("chapter", None)
    #story_id = event.get("queryStringParameters", {}).get("story", None)
    chapter_id = "1"
    story_id = "test_story"
    if not chapter_id or not story_id:
        # return error_page()
        chapter_id = "1"
        story_id = "test_story"

    # print(f"Querying for story: {story_id}")
    # response = ddbclient.query(
    #     TableName=DDB_TABLE_NAME,
    #     KeyConditionExpression="story = :story",
    #     ExpressionAttributeValues={":story": {"S": story_id}},
    # )
    response =  {'Items': [{'chapter': {'S': '1'}, 'options': {'L': [{'M': {'next': {'S': '2'}}}]}, 'text': {'S': 'Once upon a time there was a young boy. '}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/1-2.png'}}, {'chapter': {'S': '2'}, 'options': {'L': [{'M': {'next': {'S': '3'}}}]}, 'text': {'S': 'The boy lived in a small house in a large forest'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/2-7.png'}}, {'chapter': {'S': '3'}, 'options': {'L': [{'M': {'next': {'S': '3.1'}, 'text': {'S': 'The boy went outside to investigate'}}}, {'M': {'next': {'S': '3.2'}, 'text': {'S': 'The boy, who was used to these kind of sounds, quickly fell back asleep'}}}]}, 'text': {'S': 'One night, the boy woke up from a loud cracking sound just outside his house.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/3-3.png'}}, {'chapter': {'S': '3.1'}, 'options': {'L': [{'M': {'next': {'S': '-1'}, 'text': {'S': 'he fetched a flashlight inside, and then returned'}}}, {'M': {'next': {'S': '3.2'}, 'text': {'S': 'he went back to bed to wait for the morning light'}}}]}, 'text': {'S': 'The night was dark so the boy could not see anything.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/3.1-2.png'}}, {'chapter': {'S': '3.2'}, 'options': {'L': [{'M': {'next': {'S': '-1'}, 'text': {'S': 'The boy had some breakfast and then went outside to check what had made the sound'}}}]}, 'text': {'S': 'A few hours later the boy woke up to a bright morning.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/3.2-9.png'}}, {'chapter': {'S': '-1'}, 'text': {'S': 'The boy saw that the old oak tree had fallen over and inside sat a small pixie with purple eyes. Then the boy woke up.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/FINAL_CHAPTER-2.png'}}], 'Count': 6, 'ScannedCount': 6, 'ResponseMetadata': {'RequestId': '81RT0S7OISJUMUPABFMPQ61LKRVV4KQNSO5AEMVJF66Q9ASUAAJG', 'HTTPStatusCode': 200, 'HTTPHeaders': {'server': 'Server', 'date': 'Tue, 12 Jul 2022 00:21:03 GMT', 'content-type': 'application/x-amz-json-1.0', 'content-length': '2128', 'connection': 'keep-alive', 'x-amzn-requestid': '81RT0S7OISJUMUPABFMPQ61LKRVV4KQNSO5AEMVJF66Q9ASUAAJG', 'x-amz-crc32': '980735285'}, 'RetryAttempts': 0}}

    # print("Got response")
    # print(response)

    items = response.get("Items", [])
    if len(items) == 0:
        return error_page()

    chapters = {}

    for item in items:
        chapter = Chapter(chapter=from_dynamodb_to_json(item))
        chapters[chapter.chapter_id] = chapter

    nodes = get_nodes(chapters)
    edges = get_edges(chapters)


    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html"},
        "body": generate_body(get_edit_windows(chapters), nodes, edges),
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


def get_node(id, label):
    return '{ id: "%s", label: "%s" },' % (id, label)


def get_nodes(chapters: Dict[str, Chapter]):
    return "\n".join([get_node(key, key) for key in chapters.keys()])


def get_edit_window_existing_option(option_id: str, option_text: str):
    return """
        <label for="{option_id}">Choice leading to chapter: {option_id}</label>
        <input type="text" id="{option_id}" name="{option_id}" default="{option_text}" value="{option_text}"><br><br>
    """.format(
        option_id=option_id, option_text=option_text
    )


def get_edit_window(chapter: Chapter, next_choice_id: int):
    options = []
    for option_id, option_text in chapter.get_choices().items():
        options.append(get_edit_window_existing_option(option_id, option_text))
    return get_edit_window_html().format(
        CHAPTER_ID=chapter.chapter_id,
        IMAGE="", # chapter.image, # TODO remove
        EXISTING_OPTIONS="""
        """.join(options),
        STORY_ID=chapter.chapter_id,
        NEXT_CHOICE_ID=next_choice_id,
        CHAPTER_TEXT=chapter.text,
    )


def get_edit_windows(chapters: Dict[str, Chapter]):
    next_choice_id = max([int(id.split(".")[0]) for id in chapters]) + 1
    return "".join([get_edit_window(chapter, next_choice_id) for chapter in chapters.values()])


def get_edges(chapters: Dict[str, Chapter]):
    edges = []
    for chapter_id, chapter in chapters.items():
        for choice in chapter.get_choices().keys():
            edges.append(get_edge(chapter_id, choice))
    return """
    """.join(edges)


def generate_body(edit_windows, nodes, edges):
    html = get_html()
    style = get_style()
    formatted = html % (
        style, 
        edit_windows, 
        nodes, 
        edges
    )
    print(formatted)


DDB_TABLE = "arn:aws:dynamodb:eu-central-1:679585051464:table/stories"

DDB_TABLE_NAME = "stories"

LAST_CHAPTER_PREFIX = "-"

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
        height: 50%;
        border: 1px solid lightgray;
        min-height: 50%
    }
    
    center { 
        min-height: 50%
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
            var options = { };
            var network = new vis.Network(container, data, options);
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
    return  """
    <div class="edit-window" id="edit-window-{CHAPTER_ID}">
        <img src="{IMAGE}" loading="lazy">
        <form action="/process_edit_story" method="get">
            <label for="_text">Chapter text: </label>
            <input type="text" id="_text" name="_text" value="{CHAPTER_TEXT}"><br><br>
            {EXISTING_OPTIONS}
            <label for="new_option">New choice text: </label>
            <input type="text" id="new_option_text" name="new_option_text"><br><br>
            
            <label for="new_option">New choice id: </label>
            <input type="text" id="new_option_id" name="new_option_id" value="{NEXT_CHOICE_ID}"><br><br>
            
            <input type="hidden" name="story_id" value={STORY_ID} class="hidden">
            <input type="hidden" name="chapter_id" value={CHAPTER_ID} class="hidden">

            <input type="submit" value="Submit">
        </form>
    </div>
    """

if __name__=="__main__":
    lambda_handler(None, None)