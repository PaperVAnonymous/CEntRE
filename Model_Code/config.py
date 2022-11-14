#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import sys
sys.path.append('..')
from data_utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def processing_origdata(data_file):

    dataset = json.load(open(data_file))
    max_seq_length = dataset['max_length']  # the max length of sentence

    relation_dict = dataset['relations_dict']  # 关系字典，关系：关系id

    data = dataset['data']  # 每个元素是一个字典，代表了一条数据
    ent_cnt = []
    for each_data in data:
        ent_cnt.append(each_data['entities_num'])
    max_ent_cnt = max(ent_cnt)

    return max_seq_length, relation_dict, max_ent_cnt, data


class BasicArgs:
    BertPath = 'XXX/Enterprise_Relations/NER/BERT/BERT_Chinese'
    GazVec = 'XXX/Enterprise_Relations/pretrained_emb/Tencent_AILab_ChineseEmbedding.txt'
    WordVec = ''
    Original_Dataset = ''
    Bert_saved_path = ''
    JointModel_saved_path = ''

    max_seq_length, relation_dict, max_ent_cnt, data = processing_origdata(Original_Dataset)  # 记得划分train/dev/test
    data_piece = len(data) // 10
    train_data = data[: data_piece * 8]
    dev_data = data[data_piece * 8: data_piece * 9]
    test_data = data[data_piece * 9:]

    batch_size = 1
    max_seq_len = 512
    learning_rate = 5e-3
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    total_epoch = 100

    weight_decay_finetune = 1e-5
    lr_crf_fc = 1e-5
    weight_decay_crf_fc = 1e-5
    warmup_proportion = 0.002

    word_emb_dim = 200
    bert_emb_dim = 768
    hid_dim = 256

    transformer_num_layer = 4
    transformer_num_heads = 8
    transformer_mod_dim = 512
    transformer_ff_dim = 512

    unified_encoder_output_dim = 512

    semantic_num_sim = 3
    gaz_alphabet_emb_dim = 200
    dropout_rate = 0.5

    lr = 5e-4
    weight_decay = 0.001
    min_lr = 5e-5
    lr_decay_factor = 0.5

    gradient_accumulation_steps = 40

    """
    *************************************上述主要面向NER，下面面向RE。*************************************
    """
    max_ent_size = 50  # 实体的尺寸大小
    max_distance = max_seq_length  # 其他token相对实体token的最远距离

    ent_size_emb_dim = 50  # 实体的尺寸向量维度
    label_emb_dim = 25  # 实体标签的向量表示
    rel_emb_dim = 100  # 关系标签的向量表示
    relative_dis_dim = 50  # 相对位置的向量表示
    re_cat_input_dim = 637  # relative_dis_dim*2+label_emb_dim+unified_encoder_output_dim
    re_unified_input_dim = 256
    re_hid_dim = 512  # 一定要保证和unified_encoder_output_dim一致
    re_bilstm_layers = 4

    ent_unified_feature_dim = 512

    """
    *************************************下述主要面向re_mlp*************************************
    """
    mlp_input_dim = 512  # 要与re_hid_dim一致
    mlp_hid_dim = 256

