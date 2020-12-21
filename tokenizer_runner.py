import os
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tokenizer import *

def tokenize_path(tokenize_func):
    def flex(path):
        if type(path) != str:
            return path
        first, nonterminal, second = path.split(',')
        return ','.join([tokenize_func(first), nonterminal, tokenize_func(second)])
    return flex


def recount(counter, tokenize):
    new_counter = {}
    for word, count in counter.items():
        for token in tokenize(word).split('|'):
            new_counter[token] = new_counter.get(token, 0) + count
    return new_counter


def run_tokenizer(tokenizer_class, dir_name, mode='train'):
    print(dir_name)
    with open('data/java-small/java-small.dict.c2s', 'rb') as file:
        subtoken_to_count_old = pickle.load(file)
        node_to_count_old = pickle.load(file)
        target_to_count_old = pickle.load(file)
        max_contexts = pickle.load(file)
        num_training_examples = pickle.load(file)
        print('Dictionaries loaded.')
    print(len(target_to_count_old), len(subtoken_to_count_old))
    tokenize_target = tokenizer_class('target')
    tokenize_subtoken = tokenizer_class('subtoken')
    if mode == 'train':
        print('Training target')
        tokenize_target.train(target_to_count_old)
        print('Training subtoken')
        tokenize_subtoken.train(subtoken_to_count_old)
    tokenize_target.load()
    tokenize_subtoken.load()

    filename = f'data/{dir_name}/java-small.{mode}.c2s'
    Path(f'data/{dir_name}').mkdir(parents=True, exist_ok=True)
    if Path(filename).exists():
        os.remove(filename)
    chunk = 10000
    for df in tqdm(pd.read_csv(f'data/java-small/java-small.{mode}.c2s', sep=' ', header=None, chunksize=chunk),
                   total=num_training_examples // chunk + 1):
        df[0] = df[0].apply(tokenize_target)
        for i in range(1, max_contexts):
            df[i] = df[i].apply(tokenize_path(tokenize_subtoken))
        df.to_csv(filename, header=None, mode='a', index=False, sep=' ')

    if mode == 'train':
        pkl = f'data/{dir_name}/java-small.dict.c2s'
        if Path(pkl).exists():
            os.remove(pkl)
        print('Recount target')
        target_to_count = recount(target_to_count_old, tokenize_target)
        print('Recount subtoken')
        subtoken_to_count = recount(subtoken_to_count_old, tokenize_subtoken)
        with open(pkl, 'wb') as file:
            pickle.dump(subtoken_to_count, file)
            pickle.dump(node_to_count_old, file)
            pickle.dump(target_to_count, file)
            pickle.dump(max_contexts, file)
            pickle.dump(num_training_examples, file)
            print('Dictionaries saved.')


if __name__ == '__main__':
    # run_tokenizer(CamelCaseTokenizer, 'java-small-camel', 'train')
    # run_tokenizer(CamelCaseTokenizer, 'java-small-camel', 'val')
    # run_tokenizer(CamelCaseTokenizer, 'java-small-camel', 'test')
    #
    run_tokenizer(BpeTokenizer, 'java-small-bpe', 'train')
    run_tokenizer(BpeTokenizer, 'java-small-bpe', 'val')
    run_tokenizer(BpeTokenizer, 'java-small-bpe', 'test')

    run_tokenizer(UnigramTokenizer, 'java-small-unigram', 'train')
    run_tokenizer(UnigramTokenizer, 'java-small-unigram', 'val')
    run_tokenizer(UnigramTokenizer, 'java-small-unigram', 'test')