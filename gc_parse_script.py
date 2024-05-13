import gzip
from multiprocessing import Pool, cpu_count
import json
import os
from typing import Callable
import pandas as pd

def get_files_recursively(folder: str, extension: str, function: Callable,):
    """
    Iterates through a folder, and its subfolders recursively.

    folder: the main folder to traverse
    extension: the expected extension for file to be processed
    function: the (Callable) function to be executed at the selected files, needs to get a path (str)
    pattern_in_filename: regex compiled pattern, that should be found in the filename to process the file
    """
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist!")
        return
    res = []
    for root, dirs, files in os.walk(folder):
        with Pool(cpu_count) as pool:
            for file in files:
                if file.endswith(f".{extension}"):
                    parse = (pool.apply_async(function(os.path.join(root, file))))
                    res.append(parse.get())
    return pd.concat(res, ignore_index=True)

def parse_cc(infile: str):
    data = []
    with gzip.open(infile) as f:
        for line in f:
            data.append(list(unpack(json.loads(line))))
    return pd.DataFrame(data, columns=['id','index', 'domain', 'warc', 'offset', 'length', 'status', 'mime', 'response_date', 'response_content_type', 'text'])




def unpack(data):
    for k, v in data.items():
        if isinstance(v, dict):
            yield from unpack(v)
        else:
            yield v



if __name__=='__main__':
    os.nice(32)
    data_folder = "./data/orlando/Language/Hungarian/Corpus/cc/hu/"
    ext = ".gz"
    res = get_files_recursively(data_folder, ext, parse_cc)
    res.to_csv("./data/orlando/Projects/Makako/giant_corpus.csv")