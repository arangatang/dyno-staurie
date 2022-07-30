def error_page(reason: str = ""):
    return {"statusCode": 404, "body": f"{reason}"}


DDB_TABLE = "arn:aws:dynamodb:eu-central-1:679585051464:table/stories"
DDB_TABLE_NAME = "stories"
LAST_CHAPTER_PREFIX = "FINAL_CHAPTER"
DEFAULT_IMAGE = "https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/1-0.png"
