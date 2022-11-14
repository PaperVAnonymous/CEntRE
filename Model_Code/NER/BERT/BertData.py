#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn
import sys
sys.path.append('..')
from NER.structure_augmentation.alphabet import Alphabet


class BertData:
    def __init__(self):
        self.label_alphabet = Alphabet('label', True)

    def build_alphabet(self,data):
        for each_data in data:
            for label in each_data['label']:  # label_alphabet添加了<BOS>和<EOS>
                self.label_alphabet.add(label)

