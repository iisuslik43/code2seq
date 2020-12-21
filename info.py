import os
import pickle

if __name__ == '__main__':
    with open('data/java-small/java-small.dict.c2s', 'rb') as file:
        subtoken_to_count_old = pickle.load(file)
        node_to_count_old = pickle.load(file)
        target_to_count_old = pickle.load(file)
        max_contexts = pickle.load(file)
        num_training_examples = pickle.load(file)
        print('Dictionaries loaded.')
    print(len(target_to_count_old), len(subtoken_to_count_old), num_training_examples)
