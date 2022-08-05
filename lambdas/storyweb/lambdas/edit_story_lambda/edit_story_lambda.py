from typing import Dict, List
import boto3
from storyweb.utils.chapters import Chapter
from storyweb.utils.constants import DDB_TABLE_NAME, error_page
from storyweb.style import load_css
from storyweb.utils.loaders import get_jinja_environment
from storyweb.utils.utils import from_dynamodb_to_json

ddbclient = boto3.client("dynamodb")


def lambda_handler(event, context):
    chapter_id = event.get("queryStringParameters", {}).get("chapter", "1")
    story_id = event.get("queryStringParameters", {}).get("story", "test_story")

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
        "body": get_jinja_environment()
        .get_template("edit_story_template.jinja")
        .render(
            nodes=nodes,
            edges=edges,
            select_node=chapter_id if chapter_id else "",
            chapters=[chapter.to_dict() for chapter in chapters.values()],
            story_name=story_id,
            new_option_id=max([int(id.split(".")[0]) for id in chapters]) + 1,
        ),
    }


def get_nodes(chapters: Dict[str, Chapter]):
    def get_node(id: str, is_work_in_progress: bool, is_final: bool, is_start: bool):
        if is_work_in_progress:
            node_color = "#FFFF00"  # yellow
        elif is_start:
            node_color = "#00FF00"  # green
        elif is_final:
            node_color = "#FF0000"  # red
        else:
            node_color = "#4DA4EA"  # blue

        color = 'color: { background: "%s"}' % node_color
        return '{ id: "%s", label: "%s", %s },' % (id, id, color)

    nodes = []
    for key, chapter in chapters.items():
        nodes.append(
            get_node(
                key, chapter.is_unfinished(), chapter.is_final(), chapter.is_start()
            )
        )

    return "\n".join(nodes)


def get_edges(chapters: Dict[str, Chapter]):
    def get_edge(_from, to):
        return (
            '{ from: %s, to: %s, arrows: { to: { enabled: true, type: "arrow" }}},'
            % (
                _from,
                to,
            )
        )

    edges = []
    for chapter_id, chapter in chapters.items():
        for choice in chapter.get_choices().keys():
            edges.append(get_edge(chapter_id, choice))
    return """
    """.join(
        edges
    )
