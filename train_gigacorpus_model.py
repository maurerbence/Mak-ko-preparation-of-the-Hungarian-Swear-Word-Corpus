import argparse
import json
import os

from transformers import AutoTokenizer
from gensim.models import Word2Vec
from utils import MakakoCorpus


def train_on_gigacorpus(corpus, model_name_str):

    tokenized_text = []
    for _, tokens in corpus:
        tokenized_text.append(tokens)
    model = Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, sg=1)
    print("Building vocabulary...")
    model.build_vocab(corpus)
    print("Training Model...")
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

    train_on_gigacorpus(corpus, model_name_str)

    swear_text = extract_sentences_with_swear_words(corpus, corpus.swear_words)

    return swear_text


if __name__ == "__main__":
    swear_docs = main()
    with open("swear_docs.json", "w") as file:
        json.dump(swear_docs, file)
