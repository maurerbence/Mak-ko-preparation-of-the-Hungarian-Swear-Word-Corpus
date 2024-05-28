import argparse
import json
import logging
import os
import time

from transformers import AutoTokenizer
from gensim.models import Word2Vec
from utils import MakakoCorpus
from multiprocessing import cpu_count, Pool


def train_on_gigacorpus(corpus, model_name_str,):

    processes = cpu_count()
    model = Word2Vec(vector_size=300, min_count=50, workers=processes-1, epochs=1,)
    logging.info(":::::::Building vocabulary...")
    model.build_vocab(corpus)
    logging.info(":::::::Training Model...")
    model.train(corpus, total_examples=model.corpus_count, epochs=10)
    print(f"Saving model as {model_name_str}")
    model.save(model_name_str)


def contains_swear_word(tokens, swear_words):
    return any(word in tokens for word in swear_words)


def extract_sentences_with_swear_words(corpus, swear_words):
    sentences_with_swear_words = []
    for tokens, document in corpus:
        if contains_swear_word(tokens, swear_words):
            sentences_with_swear_words.append(document)
    return sentences_with_swear_words


def main():
    """ """
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()],
        level=logging.INFO)

    logger = logging.getLogger(__name__)

    os.nice(20)
    parser = argparse.ArgumentParser()
    # root_dir and model_name
    parser.add_argument("root_dir", help="Path to the input Corpus texts")
    parser.add_argument("model_name", help="Name of the model (run, batch size, etc.)")
    args = parser.parse_args()

    root_dir = args.root_dir
    model_name_str = f"{args.model_name + '.model' if not args.model_name.endswith('.model') else args.model_name}"
    puli_tokenizer = AutoTokenizer.from_pretrained("NYTK/PULI-GPT-3SX")

    corpus = MakakoCorpus(root_dir, puli_tokenizer)

    train_on_gigacorpus(corpus, model_name_str, )


if __name__ == "__main__":
    main()

