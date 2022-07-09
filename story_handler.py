from functools import cached_property
import logging
from pathlib import Path
from typing import Iterable, List
import background_handler
import yaml

LAST_CHAPTER_PREFIX = "FINAL_CHAPTER"


class ChapterOptions:
    options: List[dict]

    def __init__(self, options: List[dict]):
        self.options = options

    def get_choices(self) -> Iterable:
        return [option.get("text", "Continue") for option in self.options]

    def get_path_for_choice(self, option: int) -> str:
        logging.info(self.options)
        
        return str(self.options[option].get("next"))


class Chapter:
    text: str = ""
    options: ChapterOptions = ""
    is_final_chapter: bool = True

    def __init__(self, chapter: dict, chapter_id: str):
        # Chapter numbers start with 1.yml
        # For multiple choices subsequent
        if not chapter.get("text"):
            raise ValueError("Found chapter without text", chapter_id, chapter)
        else:
            self.text = chapter["text"]

        self.is_final_chapter = chapter_id.startswith(LAST_CHAPTER_PREFIX)

        if not self.is_final_chapter:
            # if no option given, then always set next chapter as current chapter + 1
            self.options = ChapterOptions(
                chapter.get("options", [{"text": "continue", "next": LAST_CHAPTER_PREFIX}])
            )

    def get_choices(self):
        if self.is_final_chapter:
            return []

        return self.options.get_choices()

    def get_path_for_choice(self, choice):
        return self.options.get_path_for_choice(choice)


def load_chapter(story_path: Path, chapter_name: str):
    chapter_path = story_path / f'{chapter_name}.yml'
    logging.info(f"loading chapter: {chapter_path}")
    with chapter_path.open("r") as fp:
        fc = yaml.safe_load(fp)
        return Chapter(fc, chapter_name)


def get_user_input(max_val: int):
    while True:
        answer = input("> ")
        try:
            if answer == "":
                return 0

            answer = int(answer)
            if not 0 <= answer <= max_val:
                continue
            logging.info(f"User chose answer: {answer}")
        except ValueError:
            continue
        return answer


def run_story(story_path: Path):
    current_chapter = "1"

    while True:
        chapter = load_chapter(story_path=story_path, chapter_name=current_chapter)
        print(chapter.text)

        choices = chapter.get_choices()
        for num, possible_choice in enumerate(choices):
            print(f"{num}: {possible_choice}")
        
        background_handler.generate_background(chapter.text)

        if(chapter.is_final_chapter):
            break

        answer = get_user_input(max_val=len(choices) - 1)
        current_chapter = chapter.get_path_for_choice(answer)
