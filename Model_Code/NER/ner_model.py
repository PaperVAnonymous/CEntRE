#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
sys.path.append('..')
from NER.structure_augmentation.module import SModule
from NER.BERT.BertFeature import BertFeature
from NER.a_position_emb import APosEmb
from NER.semantic_augmentation import SemanticAug
from NER.transformer import AttentionModel
from NER.crf import CRF
from data_utils import *


class NER_module(nn.Module):
    def __init__(self, basic_args, data):
        super(NER_module,self).__init__()
        self.args = basic_args

        self.smodule = SModule(data)  # for the structure augmentation
        self.feature_dim = self.smodule.feature_dim  # the word_emb_dim+4*gaz_emb_dim

        self.word_alphabet = data.word_alphabet
        self.label_alphabet = data.label_alphabet
        self.num_labels = self.label_alphabet.size()

        self.bert_module = BertFeature(self.args.BertPath, self.args.Bert_saved_path,
                                       self.label_alphabet, self.args.device)
        self.bert2unify = nn.Linear(768, self.args.unified_encoder_output_dim)

        self.semantic_aug = SemanticAug(self.args.GazVec, self.word_alphabet, self.args.gaz_alphabet_emb_dim)
        self.semantic_emb_table = nn.Embedding(self.word_alphabet.size(), self.args.gaz_alphabet_emb_dim)
        self.semantic_emb_table.weight.data.copy_(torch.from_numpy(self.semantic_aug.semantic_emb_talbe()))
        self.semantic2unify = nn.Linear(self.smodule.gaz_alphabet_emb_dim, self.args.unified_encoder_output_dim)

        self.aposition_module = APosEmb(self.feature_dim, self.args.dropout_rate)
        self.transformer_encoder = AttentionModel(self.feature_dim, self.args.transformer_mod_dim,
                                                  self.args.transformer_ff_dim, self.args.transformer_num_heads,
                                                  self.args.transformer_num_layer, self.args.dropout_rate, self.args.device)

        self.transformer2unify = nn.Linear(self.args.transformer_mod_dim, self.args.unified_encoder_output_dim)

        self.gate1_w = nn.Parameter(torch.empty(2 * self.args.unified_encoder_output_dim, self.args.unified_encoder_output_dim), requires_grad=True)
        nn.init.xavier_normal_(self.gate1_w)
        self.gate1_linear = nn.Linear(self.args.unified_encoder_output_dim, self.args.unified_encoder_output_dim)

        self.gate2_w = nn.Parameter(torch.empty(2 * self.args.unified_encoder_output_dim, self.args.unified_encoder_output_dim), requires_grad=True)
        nn.init.xavier_normal_(self.gate2_w)
        self.gate2_linear = nn.Linear(self.args.unified_encoder_output_dim, self.args.unified_encoder_output_dim)

        self.unify2emission = nn.Linear(self.args.unified_encoder_output_dim, self.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.args.dropout_rate)
        """****************The above module belongs to encoder which is for getting feature from input sequence**************"""

        self.pad_index = self.label_alphabet.get_index('<PAD>')
        self.bos_index = self.label_alphabet.get_index('<BOS>')
        self.eos_index = self.label_alphabet.get_index('<EOS>')
        self.crf = CRF(self.num_labels, self.pad_index, self.bos_index, self.eos_index, 'cuda')

    def encoder_output(self, instance):# return the fused feature after encoder
        seq_len = len(instance[0])
        structure_aug_feature = self.smodule(instance)  # the feature from structure augmentation, (seq_len, feature_dim)
        structure_aposition = self.aposition_module(structure_aug_feature, seq_len)  # (seq_len, feature_dim)
        # attention_mask = torch.tensor([0]*seq_len, dtype=torch.long).unsqueeze(0).unsqueeze(-1).expand(-1,-1, seq_len)
        # transformer_output = self.transformer_encoder(structure_aposition.unsqueeze(0), attention_mask).squeeze(0)  # (seq_len, model_dim)
        transformer_output = self.transformer_encoder(structure_aposition.unsqueeze(0)).squeeze(0)  # (seq_len, model_dim)

        return transformer_output
        
    def fused_gate(self, instance):
        transformer_unify_output = self.transformer2unify(self.encoder_output(instance))
        bert_unify_output = self.bert2unify(self.bert_module(instance[0]))  # instance[0] == the word list
        semantic_unify_output = self.semantic2unify(self.semantic_emb_table(torch.tensor(instance[1], dtype=torch.long)))

        transformer_bert_cat = torch.cat([transformer_unify_output, bert_unify_output], dim=-1)
        transformer_bert_gate = transformer_bert_cat.matmul(self.gate1_w)
        transformer_bert_gate = self.sigmoid(transformer_bert_gate)
        transformer_bert_ones = torch.ones(transformer_bert_gate.shape).to(self.args.device)
        transformer_bert_fusion = transformer_bert_gate.mul(transformer_unify_output)+\
                                  (transformer_bert_ones-transformer_bert_gate).mul(bert_unify_output)
        transformer_bert_fusion = self.gate1_linear(transformer_bert_fusion)


        twos_semantic_cat = torch.cat([transformer_bert_fusion, semantic_unify_output], dim=-1)
        twos_semantic_gate = twos_semantic_cat.matmul(self.gate2_w)
        twos_semantic_gate = self.sigmoid(twos_semantic_gate)
        twos_semantic_ones = torch.ones(twos_semantic_gate.shape).to(self.args.device)
        all_fused_feature = twos_semantic_gate.mul(transformer_bert_fusion)+\
                                  (twos_semantic_ones-twos_semantic_gate).mul(semantic_unify_output)
        all_fused_feature = self.gate2_linear(all_fused_feature)

        # fused_feature = semantic_unify_output + transformer_bert_fusion

        return self.dropout(all_fused_feature)  # this should be the fused feature with unified dim

    def uniform_prepro(self, instance):
        emission_feature = self.unify2emission(self.fused_gate(instance))  # 需要对instance数据进行分解，类似于DataConf
        true_label_ids = torch.tensor(instance[3], dtype=torch.long)
        valid_sent_mask = true_label_ids != self.pad_index

        emission_feature = emission_feature.unsqueeze(0)  # 扩充第一维成为batch_size
        true_label_ids = true_label_ids.unsqueeze(0)
        valid_sent_mask = valid_sent_mask.unsqueeze(0)

        return emission_feature, true_label_ids, valid_sent_mask


    def ner_train(self, instance):
        emission_feature, true_label_ids, valid_sent_mask = self.uniform_prepro(instance)
        # object_score = F.cross_entropy(emission_feature.view(-1, self.num_labels), true_label_ids.view(-1), ignore_index=self.pad_index)
        object_score = self.crf(emission_feature, true_label_ids, valid_sent_mask)  # come from crf file
        return object_score


    def ner_test(self, instance):
        emission_feature, true_label_ids, valid_sent_mask = self.uniform_prepro(instance)
        _ , best_path = self.crf.viterbi_decode(emission_feature, valid_sent_mask)
        return best_path[0], instance[3]  # 我们只有一条数据，只取第一个path, instance[3]是真实的标签序列

    def extract_ents(self, best_path):
        """
        :param best_path: 在test阶段，ner_test输出的标注序列
        :return: 预测输出的实体构造的列表，每个实体包含了其其实位置
        """
        ent_start = 0
        ent_list = []
        seq_len = len(best_path)
        while ent_start < seq_len:
            while ent_start < seq_len and best_path[ent_start] == self.label_alphabet.get_instance('O'):
                # 保证开始不能是'O'
                ent_start = ent_start+1
            ent_end = ent_start+1
            while ent_end < seq_len and best_path[ent_end]==self.label_alphabet.get_instance('I'):
                # 维持中间位置都是'I'
                ent_end = ent_end+1
            if ent_start != seq_len:
                ent_list.append((ent_start, ent_end))
            ent_start = ent_end

        return ent_list




