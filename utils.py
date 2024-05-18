import argparse
import gensim
import gzip
import json
import os
import xml.etree.ElementTree as et

from transformers import AutoTokenizer


def dirty_word_list():
    xml_doc = et.parse("dirtywords.xml")
    root = xml_doc._root
    words = root.findall("Word")
    # Only want to extract 1- and 2-grams for now
    word_list = [word.text.lower() for word in words if (word.text is not None and len(word.text.split(" ")) <= 2)]
    return word_list


class MakakoCorpus:
    """GigaCorpus"""
    def __init__(self, root_dir, tokenizer: AutoTokenizer):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.swear_words = dirty_word_list()

    def __iter__(self):
        for root, _, files in os.walk(self.root_dir):
            for file_name in files:
                if file_name.endswith('.jsonl.gz'):
                    file_path = os.path.join(root, file_name)
                    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
                        for line in file:
                            data = json.loads(line)
                            document = data.get('text', '')
                            if document:
                                tokens = self.tokenizer.tokenize(document)
                                yield document, tokens

    @staticmethod
    def contains_swear_word(text: str, swear_words: list):
        return any(word in text for word in swear_words)


# Planning to use as module.
if __name__ == "__main__":
    print(dirty_word_list())
