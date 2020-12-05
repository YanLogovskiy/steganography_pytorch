import re
import random


class TextSplitter:
    def __init__(self, path, interval=(150, 210)):
        text_file = open(path, 'r')
        parsed_text = re.sub(r'\s+', ' ', text_file.read(), flags=re.M)
        self.sentences = [sentence for sentence in re.split(r'(?<=[.!?…]) ', parsed_text)
                          if (interval[0] < len(sentence) < interval[1])]

    def __call__(self):
        return random.choice(self.sentences)
