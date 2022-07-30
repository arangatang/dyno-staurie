import click
from storyweb import edit_handler, read_handler, update_handler
from storyweb.utils.local_development import dump


@click.group()
def cli():
    pass


@cli.command()
def edit():
    page = edit_handler({}, {})
    dump(page)


@cli.command()
def read():
    page = read_handler({}, {})
    dump(page)


@cli.command()
def update():
    page = update_handler(
        {
            "isLocal": True,
            "queryStringParameters": {
                "chapter_id": "1",
                "chapter_text": "test chapter text",
                "story_id": "test_story",
                "new_option_id": "-123",
                "new_option_text": "some great new option text",
                "option_1": "smth",
                "option_2": "smth",
                "option_3": "smth",
                "option_4": "smth",
                "option_5": "smth",
                "option_6": "smth",
            },
        },
        {},
    )
    dump(page)
