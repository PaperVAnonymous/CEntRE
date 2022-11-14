#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import datetime
import numpy as np
import json
from numpy import random
from collections import Counter

def time_format(time_diff):
    seconds = int(round(time_diff))
    return str(datetime.timedelta(seconds = seconds))

def sentence_padding(sentences, sent_maxlen, padding_value):  # 此处的输入是batch_size数据，并经过了text2ids
    padded = []
    for sent in sentences:
        if len(sent) < sent_maxlen:
            padded.append(sent + [padding_value]*(sent_maxlen-len(sent)))
        else:
            padded.append(sent[:sent_maxlen])
    return padded


def gen_batch(data, batch_size):  # 此处的输入是全部的原始数据data，返回的也是一个batch的原始类型的data
    data = np.array(data)
    data_idx = np.arange(len(data))
    random.shuffle(data_idx)
    i = 0
    while True:
        if i + batch_size >= len(data):
            batch_idx = data_idx[i:]
            yield data[batch_idx]
            break
        else:
            batch_idx = data_idx[i: i+batch_size]
            yield data[batch_idx]
            i = i + batch_size

def random_index(data_num, batch_size):  # 此处的输入是全部的原始数据data，返回的也是一个batch的原始类型的data
    data_idx = np.arange(data_num)
    random.shuffle(data_idx)
    i = 0
    while True:
        if i + batch_size >= data_num:
            batch_idx = data_idx[i:]
            yield batch_idx
            break
        else:
            batch_idx = data_idx[i: i+batch_size]
            yield batch_idx
            i = i + batch_size
