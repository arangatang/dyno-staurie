from typing import Dict, List


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
    is_work_in_progress: bool = True
    is_start_node: bool = True
    chapter_id = ""
    image = ""
    chapter = {}

    def __init__(self, chapter: dict, filter_choices: bool = False):
        # Chapter numbers start with 1.yml
        # For multiple choices subsequent
        self.chapter = chapter
        self.text = chapter.get("text", None)
        self.filter_choices = filter_choices

        self.chapter_id = chapter["chapter"]
        self.image = chapter.get("image", "")
        self.options = ChapterOptions(chapter.get("options", []))

        self.is_final_chapter = not self.options.has_options()
        self.is_work_in_progress = chapter.get("is_work_in_progress", False)
        self.is_start_node = chapter.get("is_start_node", False)
        self.disabled_options = chapter.get("disabled_options", [])

    def get_choices(self) -> Dict[str, str]:
        if self.is_final_chapter:
            return {}
        if self.filter_choices:
            return {
                k: v
                for k, v in self.options.get_choices().items()
                if k not in self.disabled_options
            }
        else:
            return {k: v for k, v in self.options.get_choices().items()}

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
