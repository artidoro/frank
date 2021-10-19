import argparse
import csv
import json
from typing import Dict
import spacy 
from tqdm import tqdm

from constituency_parser import ParseTree


class ParsedDataset(object):
    def __init__(self, tokenizer_name):
        self.parse_trees: Dict[str, str] = {}
        self.parser = ParseTree(tokenizer_name=tokenizer_name)
        self.nlp = spacy.load("en_core_web_sm")

    def read_and_store_from_tsv(self, input_file_name, output_file_name):
        hashes_seen = set()
        with open(output_file_name, 'w') as output_file:
            with open(input_file_name, 'r') as open_file:
                data = json.load(open_file)
                for count, elt in tqdm(enumerate(data)):
                    new_elt = dict(**elt)
                    for text_name in ['summary', 'article']:
                        if text_name == 'article':
                            if elt['hash'] in hashes_seen:
                                continue
                            else:
                                hashes_seen.add(elt['hash'])
                        sents = [sent.text for sent in self.nlp(elt[text_name]).sents]
                        new_elt[f'{text_name}_sentences'] = sents
                        parse_trees, nt_idx_matrices = [], []
                        for sent_text in sents:
                            parse_tree, nt_idx_matrix = self.parser.get_parse_tree_for_raw_sent(raw_sent=sent_text)
                            parse_trees.append(parse_tree)
                            nt_idx_matrices.append(nt_idx_matrix)
                        new_elt['parse_tree'] = parse_trees
                        new_elt['nt_idx_matrices'] = nt_idx_matrices
                        json.dump(new_elt, output_file)
                        output_file.write('\n')
                    if count + 1 >= 10:
                        break
        return

    # def store_parse_trees(self, output_file):
    #     with open(output_file, 'w') as open_file:
    #         json.dump(self.parse_trees, open_file)
    #     return


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_file", default=None, type=str, required=True,
                        help="The input data file. Should be in the FRANK format.")

    parser.add_argument("--tokenizer_name", default='roberta-base', type=str,
                        help="Tokenizer name")

    args = parser.parse_args()
    parsed_data = ParsedDataset(tokenizer_name=args.tokenizer_name)

    # Read input files from folder
    input_file_name = args.data_file + '.json'
    output_file_name = args.data_file + '_with_parse.json'
    parsed_data.read_and_store_from_tsv(input_file_name=input_file_name,
                                        output_file_name=output_file_name)


if __name__ == "__main__":
    main()