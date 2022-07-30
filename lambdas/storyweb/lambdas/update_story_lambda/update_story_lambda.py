import boto3
from boto3.dynamodb.types import TypeDeserializer
from storyweb.utils.chapters import Chapter
from storyweb.utils.constants import DDB_TABLE_NAME, error_page
from storyweb.style import load_css

ddbclient = boto3.client("dynamodb")
table = boto3.resource("dynamodb").Table(DDB_TABLE_NAME)
sqs = boto3.resource("sqs")
queue = sqs.get_queue_by_name(QueueName="stories-texts-to-process.fifo")


def lambda_handler(event, context):
    params = event.get("queryStringParameters", {})
    if not (params):
        return error_page("Invalid get parameters")

    chapter_id = params.get("chapter_id", None)
    chapter_text = params.get("chapter_text", None)
    story_id = params.get("story_id", None)

    if event.get("isLocal", False):
        return generate_response(story_id, chapter_id, "123")

    if not chapter_text or not chapter_id or not story_id:
        return error_page(reason="text, chapter id and story id are required")

    new_option_id = params.get("new_option_id", None)
    new_option_text = params.get("new_option_text", None)

    existing_options = {
        k.replace("option_", ""): v
        for k, v in params.items()
        if k.startswith("option_")
    }

    for i in (
        chapter_id,
        story_id,
        chapter_text,
        new_option_id,
        new_option_text,
        *existing_options.keys(),
        *existing_options.values(),
    ):
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
            return error_page(
                reason=f"passed key {key} does not match the data stored for the chapter: {existing_options.keys()}"
            )

    # Create a new chapter if new_option_text is set and no existing chapter with that id exists
    if new_option_text and new_option_id:
        if not isinstance(
            get_chapter(story_id=story_id, chapter_id=new_option_id), Chapter
        ):
            # Upload new chapter before updating existing chapter to avoid dead nodes
            table.put_item(
                Item=dict(
                    story=story_id, chapter=new_option_id, is_work_in_progress=True
                )
            )
        disabled_options.append(new_option_id)
        existing_options[new_option_id] = new_option_text

    # Create the updated chapter
    text_was_updated = chapter_text != chapter.text
    text_to_send = chapter_text if text_was_updated else chapter.text
    ddb_item = dict(
        story=story_id,
        chapter=chapter_id,
        image=chapter.image,
        text=text_to_send,
    )

    if existing_options:
        ddb_item["options"] = [
            {"next": key, "text": value} for key, value in existing_options.items()
        ]
    if chapter.is_start():
        ddb_item["is_start_node"] = True
    if disabled_options:
        ddb_item["disabled_options"] = disabled_options

    # Upload the updated chapter
    print(f"Uploading chapter: {ddb_item}")
    table.put_item(Item=ddb_item)

    try:
        if text_was_updated:
            print("text was updated, attempting to send message to SQS")
            queue.send_message(
                MessageBody=text_to_send,
                MessageAttributes={
                    "story": {"StringValue": story_id, "DataType": "String"},
                    "chapter": {"StringValue": chapter_id, "DataType": "String"},
                },
                MessageGroupId=f"{story_id}#{chapter_id}",
            )
    except Exception as e:
        print(
            f"Unable to send message to queue: {text_to_send} chapter: {chapter_id} story_id: {story_id}, exception was: {e}"
        )
    # Return HTML with 2 buttons, one to view the new chapter and one to return to the edit screen for the newly created chapter
    return generate_response(
        story_id,
        chapter_id,
        new_option_id if new_option_text else None,
    )


def generate_response(story_id, chapter_id, new_option_id):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html"},
        "body": generate_body(
            story_id=story_id, chapter_id=chapter_id, new_option_id=new_option_id
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


def from_dynamodb_to_json(item):
    d = TypeDeserializer()
    return {k: d.deserialize(value=v) for k, v in item.items()}


def generate_body(story_id, chapter_id, new_option_id):
    html = get_html()
    style = load_css("update_story_lambda")

    new_chapter_button = (
        f'<div class=center><a id="edit_new_chapter_button class="button" href=https://bne1jvubt0.execute-api.eu-central-1.amazonaws.com/default/edit?story={story_id}&chapter={new_option_id}><center>Edit the newly created chapter</center></a></div>'
        if new_option_id
        else ""
    )
    view_updated_chapter_button = f'<div class=center><a id="view_edited_chapter_button class="button" href=https://bne1jvubt0.execute-api.eu-central-1.amazonaws.com/default/story-handler?story={story_id}&chapter={chapter_id}><center>View your changes to chapter {chapter_id}</center></a></div>'
    buttons = [new_chapter_button, view_updated_chapter_button]
    formatted = html % (style, "".join(buttons))
    # print(formatted)
    return formatted


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
