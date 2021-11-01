#%%
# compute and store constituencey parses.
from typing import Dict, List

import benepar
import numpy as np

from transformers import RobertaTokenizer, XLNetTokenizer, DistilBertTokenizer
from nltk.tree import ParentedTree


class ParseTree():
    def __init__(self, tokenizer_name, cached_parses=None):
        self.parser = benepar.Parser('benepar_en3')
        if 'roberta' in tokenizer_name:
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        if 'xlnet' in tokenizer_name:
            self.tokenizer = XLNetTokenizer.from_pretrained(tokenizer_name)
        if 'distilbert' in tokenizer_name:
            self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.cached_parses = cached_parses
        self.TREE_HEIGHT = 0
        self.NGRAM_LIMIT = 1000
        self.TOKEN_LIMIT = 250

    @staticmethod
    def remove_non_ascii(text):
        return ''.join([i if ord(i) < 128 else '' for i in text])

    def traverse_and_store(self,
                           tree: ParentedTree,
                           parse_tree_stored: List[Dict]):

        label = tree.label()
        words = [x.split('_')[0] for x in tree.leaves()]
        indices = [int(x.split('_')[-1]) for x in tree.leaves()]
        ngram_info = len(words)
        words = " ".join(words)

        if tree.height() > self.TREE_HEIGHT and ngram_info < self.NGRAM_LIMIT:
            parse_tree_stored.append({'phrase_label': label,
                                       'phrase': words,
                                       'ngram': ngram_info,
                                       'indices': indices})
        for subtree in tree:
            if type(subtree) == ParentedTree:
                self.traverse_and_store(tree=subtree, parse_tree_stored=parse_tree_stored)

        return parse_tree_stored

    def add_indices_to_terminals(self, tree):
        for idx, _ in enumerate(tree.leaves()):
            tree_location = tree.leaf_treeposition(idx)
            non_terminal = tree[tree_location[:-1]]
            non_terminal[0] = non_terminal[0] + "_" + str(idx)
        return tree

    def get_parse_tree(self, tokenized_sent):
        combined_text = [self.remove_non_ascii(x) for x in tokenized_sent]
        joined_text = " ".join(combined_text)

        if joined_text in self.cached_parses:
            parsed_tree = self.cached_parses[joined_text]
            parsed_tree = ParentedTree.fromstring(parsed_tree)
            return parsed_tree

        # tokenized_sent = self.tokenizer.tokenize(line[0])
        combined_text = [self.remove_non_ascii(x) for x in tokenized_sent]
        parsed_tree = self.parser.parse(sentence=combined_text)
        parsed_tree = ParentedTree.convert(parsed_tree)
        parsed_tree = self.add_indices_to_terminals(parsed_tree)
        return parsed_tree

    def get_parse_tree_for_raw_sent(self, raw_sent):
        tokenized_sent = self.tokenizer.tokenize(raw_sent)
        combined_text = [self.remove_non_ascii(x) for x in tokenized_sent]
        combined_text = combined_text[:self.TOKEN_LIMIT]

        parsed_tree = self.parser.parse(sentence=combined_text)
        parsed_tree = ParentedTree.convert(parsed_tree)
        parsed_tree = self.add_indices_to_terminals(parsed_tree)
        parsed_tree_as_list = self.traverse_and_store(parsed_tree, parse_tree_stored=[])

        num_tokens = len(combined_text)
        parsed_tree_as_list = self.get_one_hot_encoded_vector(parse_tree_list=parsed_tree_as_list,
                                                              num_tokens=num_tokens)

        nt_idx_matrix = [x['onehot'] for x in parsed_tree_as_list]
        return parsed_tree_as_list, nt_idx_matrix

    def get_one_hot_encoded_vector(self, parse_tree_list, num_tokens):
        for item in parse_tree_list:
            onehot_array = np.zeros((num_tokens,1))
            onehot_array[item['indices']] = 1.0
            item['onehot'] = np.squeeze(onehot_array, axis=1).tolist()
        return parse_tree_list
