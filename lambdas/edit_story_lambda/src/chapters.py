
from typing import Dict, List

from constants import LAST_CHAPTER_PREFIX


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
