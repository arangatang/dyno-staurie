from cmath import log
from email.policy import default
from pathlib import Path
import click
import logging

import story_handler
import background_handler

def main():
    story = "C:/Users/leona/Desktop/stories/dyno-staurie/stories/test_story_1"
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Executing story "+ story)
    story_handler.run_story(Path(story))

if __name__=="__main__":
    main()