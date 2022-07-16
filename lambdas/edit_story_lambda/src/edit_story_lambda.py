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
    
    def has_options(self) -> bool:
        return len(self.options) > 0


class Chapter:
    text: str = ""
    options: ChapterOptions = ""
    is_final_chapter: bool = True
    chapter_id = ""
    image = ""

    def __init__(self, chapter: dict):
        self.is_work_in_progress = chapter.get("is_work_in_progress", False)
        self.text = chapter.get("text", "")
        self.chapter_id = chapter["chapter"]
        self.image = chapter.get("image", "")
        self.options = ChapterOptions(chapter.get("options", []))

        self.is_final_chapter = not self.options.has_options()
        self.is_start_node = chapter.get("is_start_node", False)

    def get_choices(self) -> Dict[str, str]:
        if self.is_final_chapter:
            return {}
        return self.options.get_choices()

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
    chapter_id = event.get("queryStringParameters", {}).get("chapter", None)
    story_id = event.get("queryStringParameters", {}).get("story", None)
    # chapter_id = "1"
    # story_id = "test_story"
    print(chapter_id, story_id)
    if not chapter_id or not story_id:
        return error_page()
        # chapter_id = "3"
        # story_id = "test_story"

    print(f"Querying for story: {story_id}")
    response = ddbclient.query(
        TableName=DDB_TABLE_NAME,
        KeyConditionExpression="story = :story",
        ExpressionAttributeValues={":story": {"S": story_id}},
    )
    # response =  {'Items': [{'chapter': {'S': '1'}, 'options': {'L': [{'M': {'next': {'S': '2'}}}]}, 'text': {'S': 'Once upon a time there was a young boy. '}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/1-2.png'}}, {'chapter': {'S': '2'}, 'options': {'L': [{'M': {'next': {'S': '3'}}}]}, 'text': {'S': 'The boy lived in a small house in a large forest'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/2-7.png'}}, {'chapter': {'S': '3'}, 'options': {'L': [{'M': {'next': {'S': '3.1'}, 'text': {'S': 'The boy went outside to investigate'}}}, {'M': {'next': {'S': '3.2'}, 'text': {'S': 'The boy, who was used to these kind of sounds, quickly fell back asleep'}}}]}, 'text': {'S': 'One night, the boy woke up from a loud cracking sound just outside his house.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/3-3.png'}}, {'chapter': {'S': '3.1'}, 'options': {'L': [{'M': {'next': {'S': '-1'}, 'text': {'S': 'he fetched a flashlight inside, and then returned'}}}, {'M': {'next': {'S': '3.2'}, 'text': {'S': 'he went back to bed to wait for the morning light'}}}]}, 'text': {'S': 'The night was dark so the boy could not see anything.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/3.1-2.png'}}, {'chapter': {'S': '3.2'}, 'options': {'L': [{'M': {'next': {'S': '-1'}, 'text': {'S': 'The boy had some breakfast and then went outside to check what had made the sound'}}}]}, 'text': {'S': 'A few hours later the boy woke up to a bright morning.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/3.2-9.png'}}, {'chapter': {'S': '-1'}, 'text': {'S': 'The boy saw that the old oak tree had fallen over and inside sat a small pixie with purple eyes. Then the boy woke up.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/FINAL_CHAPTER-2.png'}}], 'Count': 6, 'ScannedCount': 6, 'ResponseMetadata': {'RequestId': '81RT0S7OISJUMUPABFMPQ61LKRVV4KQNSO5AEMVJF66Q9ASUAAJG', 'HTTPStatusCode': 200, 'HTTPHeaders': {'server': 'Server', 'date': 'Tue, 12 Jul 2022 00:21:03 GMT', 'content-type': 'application/x-amz-json-1.0', 'content-length': '2128', 'connection': 'keep-alive', 'x-amzn-requestid': '81RT0S7OISJUMUPABFMPQ61LKRVV4KQNSO5AEMVJF66Q9ASUAAJG', 'x-amz-crc32': '980735285'}, 'RetryAttempts': 0}}

    print("Got response")
    print(response)

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
        "body": generate_body(get_edit_windows(chapters, story_id=story_id, current_chapter=chapter_id), nodes, edges, chapter_id=chapter_id),
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


def get_node(id: str, is_work_in_progress: bool, is_final: bool, is_start: bool):   
    if is_work_in_progress:
        node_color = "#FFFF00" # yellow
    elif is_start:
        node_color = "#00FF00" # green
    elif is_final:
        node_color = "#FF0000" # red
    else:
        node_color = "#4DA4EA" # blue
        
    
    color = 'color: { background: "%s"}' % node_color
    return '{ id: "%s", label: "%s", %s },' % (id, id, color)


def get_nodes(chapters: Dict[str, Chapter]):
    nodes = []
    for key, chapter in chapters.items():
        nodes.append(get_node(key, chapter.is_unfinished(), chapter.is_final(), chapter.is_start()))
        
    return "\n".join(nodes)


def get_edit_window_existing_option(option_id: str, option_text: str):
    return """
        <label for="{option_id}">Choice leading to chapter: {option_id}</label>
        <input type="text" id="option_{option_id}" name="option_{option_id}" default="{option_text}" value="{option_text}"><br><br>
    """.format(
        option_id=option_id, option_text=option_text
    )


def get_edit_window(chapter: Chapter, next_choice_id: int, story_id: str, should_show: bool):
    options = []
    for option_id, option_text in chapter.get_choices().items():
        options.append(get_edit_window_existing_option(option_id, option_text))
        
    return get_edit_window_html().format(
        CHAPTER_ID=chapter.chapter_id,
        IMAGE= chapter.image, # TODO remove
        EXISTING_OPTIONS="""
        """.join(options),
        STORY_ID=story_id,
        NEXT_CHOICE_ID=next_choice_id,
        CHAPTER_TEXT=chapter.text,
        DISPLAY=' edit-window-clicked" style="display: block' if should_show else ''
    )


def get_edit_windows(chapters: Dict[str, Chapter], story_id: str, current_chapter: str):
    next_choice_id = max([int(id.split(".")[0]) for id in chapters]) + 1
    return "".join([get_edit_window(chapter, next_choice_id, story_id=story_id, should_show= current_chapter == chapter_id) for chapter_id, chapter in chapters.items()])


def get_edges(chapters: Dict[str, Chapter]):
    edges = []
    for chapter_id, chapter in chapters.items():
        for choice in chapter.get_choices().keys():
            edges.append(get_edge(chapter_id, choice))
    return """
    """.join(edges)

def select_node(node_id):
    return node_id if node_id else ""

def generate_body(edit_windows, nodes, edges, chapter_id):
    html = get_html()
    style = get_style()
    formatted = html % (
        style, 
        edit_windows, 
        nodes, 
        edges,
        select_node(chapter_id)
    )
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
    return  """
    <div class="edit-window{DISPLAY}" id="edit-window-{CHAPTER_ID}">
        <img src="{IMAGE}" loading="lazy">
        <form action="update_story" method="get">
            <label for="chapter_text">Chapter text: </label>
            <input type="text" id="chapter_text" name="chapter_text" value="{CHAPTER_TEXT}"><br><br>
            {EXISTING_OPTIONS}
            <label for="new_option">New choice text: </label>
            <input type="text" id="new_option_text" name="new_option_text"><br><br>
            <label for="new_option_id">The new choice should go to chapter with id: </label>
            <input type="text" id="new_option_id" name="new_option_id" value="{NEXT_CHOICE_ID}"><br><br>
            <input type="hidden" name="story_id" value={STORY_ID} class="hidden">
            <input type="hidden" name="chapter_id" value={CHAPTER_ID} class="hidden">

            <input type="submit" value="Submit">
        </form>
    </div>
    """

if __name__=="__main__":
    lambda_handler(None, None)