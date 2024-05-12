prompt_template = "Ingredient: {ing}\nClassify this ingredient IN TWO WORDS as harmful, not harmful, controversial, or unknown based on its effect on human health. Do not output more than TWO WORDS."

class OutputParser():
    def __init__(self, max_new_tokens) -> None:
        self.max_new_tokens = max_new_tokens

    def process_llm_output(self, text):
        """parse LLM outputs using the last (5 * max_new_tokens) characters
        ref: https://news.ycombinator.com/item?id=35841781"""
        if text[-(5 * self.max_new_tokens):].find("not safe") >= 0:
            return "harmful"
        if text[-(5 * self.max_new_tokens):].find("safe") >= 0:
            return "not harmful"
        elif text[-(5 * self.max_new_tokens):].find("not harmful") >= 0:
            return "not harmful"
        elif text[-(5 * self.max_new_tokens):].find("harmful") >= 0:
            return "harmful"
        elif text[-(5 * self.max_new_tokens):].find("controversial") >= 0:
            return "controversial"
        else:
            return "unknown"
