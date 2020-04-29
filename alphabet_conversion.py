import json
import random
import numpy as np
import torch
from torch.autograd import Variable

class Alphabet_Converter(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.start_idx = len(self.char2idx)
        self.max_length = 0
        self.word_list = []
        
    def alphabet_to_category(self, dataset):
        for word in dataset:
            self.word_list.append(word)
            if self.max_length < len(word):
                self.max_length = len(word)
        
        all_chars = sorted(set([w for word in dataset for w in word]))
        for char in all_chars:
            self.char2idx[char] = self.start_idx
            self.idx2char[self.start_idx] = char
            self.start_idx += 1
                
    def word_to_indices(self, word):
        index_word = []
        for char in [char for char in word]:
            index_word.append(self.char2idx[char])
        return index_word                              
                                      
    def indices_to_word(self, indices):
        word = ""
        for idx in indices:
            char = self.idx2char[idx]
            word += char
        return word
    
    def __str__(self):
        str = "Vocab information:\n"
        for idx, char in self.idx2char.items():
            str += "Char: %s Index: %d\n" % (char, idx)
        return str