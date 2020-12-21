import os
import random
from typing import Optional

import sentencepiece as spm


class BasicTokenizer():

    def tokenize(self, s):
        return s

    def train(self, counter):
        pass

    def __call__(self, s):
        return self.tokenize(s)

    def load(self):
        pass


class CamelCaseTokenizer(BasicTokenizer):
    def tokenize(self, s):
        if type(s) != str:
            return s
        splitted = s.split('|')
        res = splitted[0]
        for next_s in splitted[1:]:
            res += next_s.capitalize()
        splitted = [res]
        return '|'.join(splitted)


class SpmTokenizer(BasicTokenizer):
    def __init__(self, name, mode):
        self.model: Optional[spm.SentencePieceProcessor] = None
        self.mode = mode
        self.model_name = f'tokenizer_{mode}_{name}'

    def tokenize(self, s):
        if type(s) != str:
            return s
        if self.model is None:
            raise Exception('Model hasn\'t been loaded')
        else:
            tokens_camel = s.split('|')
            tokens_encoded = [self.model.encode_as_pieces(t) for t in tokens_camel]
            all_tokens = [t for l in tokens_encoded for t in l]
            return '|'.join(all_tokens)

    def train(self, counter):
        words = []
        for word, count in counter.items():
            words += [word] * count
        random.shuffle(words)
        with open('data/tempfile.txt', mode='a') as f:
            size = 50
            for i in range(len(words) // size):
                sentence = ' '.join(words[i * size: i * size + size]) + '\n'
                f.write(sentence)
        spm.SentencePieceTrainer.train(f'--input=data/tempfile.txt  --model_prefix={self.model_name} '
                                       f'--vocab_size=1000 --model_type={self.mode} --train_extremely_large_corpus=true')
        os.remove('data/tempfile.txt')

    def load(self):
        self.model = spm.SentencePieceProcessor(model_file=f'{self.model_name}.model')


class BpeTokenizer(SpmTokenizer):
    def __init__(self, name):
        super().__init__(name, 'bpe')


class UnigramTokenizer(SpmTokenizer):
    def __init__(self, name):
        super().__init__(name, 'unigram')