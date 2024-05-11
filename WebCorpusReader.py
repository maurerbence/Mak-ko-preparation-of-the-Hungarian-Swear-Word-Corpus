import os
import gzip
import pandas as pd
from multiprocessing import Pool, cpu_count, freeze_support
import glob
import json

class Parser:
    def parse(self, lines):
        raise NotImplementedError("Each Parser must implement the parse method.")

class ConlluParser(Parser):
    def parse(self, lines):
        data = []
        for line in lines:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split("\t")
                if len(parts) == 5:
                    data.append(parts)
        return pd.DataFrame(data, columns=["form", "wsafter", "anas", "lemma", "xpostag"])

class ConlluFileProcessor:
    def __init__(self, directory, parser):
        self.directory = directory
        self.parser = parser
    
    def _process_file(self, file_path):
        lines = []
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            skip_header = True
            for line in file:
                if skip_header:
                    if line.startswith("form\twsafter"):
                        skip_header = False
                    continue
                lines.append(line)
        
        return self.parser.parse(lines)

    def process_files(self):
        files = glob.glob(os.path.join(self.directory, '*.gz'))
        results = []
        with Pool(cpu_count()) as pool:
            for file_path in files:
                # Process each file in parallel but handle line processing sequentially
                file_result = pool.apply_async(self._process_file, (file_path,))
                results.append(file_result.get())
        return pd.concat(results, ignore_index=True)

class GigaCorpusParser(Parser):
    def parse(self, lines):
        data=[]
        for line in lines:
            data.append(list(unpack(json.loads(line))))
        return pd.DataFrame(data, columns=['id','index', 'domain', 'warc', 'offset', 'length', 'status', 'mime', 'response_date', 'response_content_type', 'text'])

class JsonlFileProcessor:
    def __init__(self, dir, parser):
        self.directory = dir
        self.parser = parser

    def _process_file(self, file):
        lines = []
        with gzip.open(file, "rt", encoding="utf-8") as f:
            for line in f:
                lines.append(line)
        return self.parser.parse(lines)

    def process_files(self):
        files = glob.glob(os.path.join(self.directory, '*.gz'))
        results = []
        with Pool(cpu_count()) as pool:
            for file in files:
                file_result = pool.apply_async(self._process_file, (file, ))
                results.append(file_result.get())
        return pd.concat(results, ignore_index=True)


def unpack(data):
    for k, v in data.items():
        if isinstance(v, dict):
            yield from unpack(v)
        else:
            yield v

