#!/usr/bin/python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append('..')
from RelE.Biaffine import Biaffine
from RelE.re_reader import InputReader
from RelE.r_position_emb import RPositionEmb
from RelE.re_encoder import Re_encoder
from RelE.sampling import *

def get_token(h, x, token): # 获取具体token的上下文表示
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    token_h = token_h[flat == token, :]
    return token_h  # (batch_size, emb_size)


class RE_module(nn.Module):
    def __init__(self, conf, label_alphabet):
        super(RE_module, self).__init__()
        self.args = conf
        self.device = conf.device
        self.label_alphabet = label_alphabet
        self.index_OBI = [self.label_alphabet.get_index('O'), self.label_alphabet.get_index('B'), self.label_alphabet.get_index('I')]

        # 面向位置对象
        self.rel_position = RPositionEmb(self.args)
        self.dis_emb = self.rel_position.position_emb

        # 面向reader对象
        self.input_reader = InputReader(self.args)
        self.rel_type_dict = self.input_reader.rel_type_dict
        self.Rel_emb = nn.Embedding(len(self.rel_type_dict), self.args.rel_emb_dim)  # 构建关系标签的emb_table
        nn.init.xavier_normal_(self.Rel_emb.weight)
        self.id2rel = self.input_reader.index2rel_type

        self.re_encoder = Re_encoder(self.args, self.label_alphabet, self.dis_emb)

        self.re_biaffine = Biaffine(self.args.ent_unified_feature_dim, self.args.ent_unified_feature_dim, len(self.rel_type_dict))  # 参数待填充
        self.dropout = nn.Dropout(conf.dropout_rate)

        self.score_softmax = nn.Softmax(dim=-1)

    def re_train(self, instance_data, hidden_emb):
        """
        :param instance_data: functions()中每条数据的最后一个[-1]，即原始数据
        :param hidden_emb:
        :return: a constant which is the loss for relation
        """
        # 构造每条数据的真实数据对象
        instance_dataset = self.input_reader.read(instance_data)

        #
        seq_len = len(instance_data['token'])  # 输入的该条数据中，最后一个
        instance_train_dict = create_train_sample(seq_len, instance_dataset, self.rel_position, self.index_OBI, self.device)
        ent1s_biaffine_input, ent2s_biaffine_input = self.re_encoder.encoder(instance_train_dict['ent_sizes'], instance_train_dict['ent_index_pair'],
                                      instance_train_dict['ent_pair_span'], instance_train_dict['ent_pair_mask'],
                                      instance_train_dict['ent_pair_tags'], instance_train_dict['ent_pair_pos'], hidden_emb)

        pairs_score = self.re_biaffine(ent1s_biaffine_input, ent2s_biaffine_input) # (rel_num, reltype_num)
        normed_pairs_score = self.score_softmax(pairs_score)

        rel_logits = F.cross_entropy(normed_pairs_score, instance_train_dict['rel_truth'])

        return rel_logits/normed_pairs_score.shape[0]  # 进行规范化

    def re_test(self, instance_data, hidden_emb, pred_ents):
        """
        :param instance_data: 输入的原始数据
        :param hidden_emb: 输入的编码后的隐层向量
        :param pred_ents: 预测出来的实体列表[(start, end), ...]
        :return: pred的关系列表以及true关系列表
        """

        instance_dataset = self.input_reader.read(instance_data)
        seq_len = len(instance_data['token'])
        instance_true_dict = create_train_sample(seq_len, instance_dataset, self.rel_position, self.index_OBI, self.device)

        instance_pred_dict = create_test_sample(seq_len, pred_ents, self.rel_position, self.index_OBI, self.device)
        test_ent1s_biaffine_input, test_ent2s_biaffine_input = self.re_encoder.encoder(instance_pred_dict['ent_sizes'], instance_pred_dict['test_ent_index_pair'],
                instance_pred_dict['test_ent_pair_span'], instance_pred_dict['test_ent_pair_mask'],
                instance_pred_dict['test_ent_pair_tags'], instance_pred_dict['test_ent_pair_pos'], hidden_emb)
        test_pairs_score = self.re_biaffine(test_ent1s_biaffine_input, test_ent2s_biaffine_input)  # (rel_num, reltype_num)
        test_normed_pairs_score = self.score_softmax(test_pairs_score)
        test_rel_label = test_normed_pairs_score.argmax(dim=-1).tolist()

        assert len(instance_pred_dict['test_ent_pair_span']) == len(test_rel_label)  # 输入的实体对和预测出的关系数量一致
        pred_ent_pair_span = instance_pred_dict['test_ent_pair_span'].tolist()  # 预测出来的实体构成的所有实体对, (pre_rel_num, 2, 2)
        pred_rel_triplets = []
        for i, pred_label in enumerate(test_rel_label):
            test_head = pred_ent_pair_span[i][0] # 二元列表，第一个实体的始末位置
            test_tail = pred_ent_pair_span[i][1] # 二元列表，第二个实体的始末位置
            test_convert_rel = self.adjust_rel((test_head, test_tail, pred_label))

            if test_convert_rel:
                pred_rel_triplets.append(test_convert_rel)


        true_rel_label = instance_true_dict['rel_truth'].tolist()
        true_ent_pair_span = instance_true_dict['ent_pair_span'].tolist()
        true_rel_triplets = []
        for j, true_label in enumerate(true_rel_label):
            true_head = true_ent_pair_span[j][0]
            true_tail = true_ent_pair_span[j][1]
            true_convert_rel = self.adjust_rel((true_head, true_tail, true_label))

            if true_convert_rel:
                true_rel_triplets.append(true_convert_rel)  # this is the groundtruth for this data

        print("Begin the instance evaluation !\n")
        self.evaluation(pred_rel_triplets, true_rel_triplets)

        return pred_rel_triplets, true_rel_triplets

    def adjust_rel(self, rel_triplet):
        """
        用于规范化实体顺序(存在对称关系的实体，索引小的在前面)，过滤掉无关系实体对
        """
        rel_label = rel_triplet[-1]
        adjusted_rel = rel_triplet
        if self.id2rel[rel_label].get_name == 'None':
            return None
        elif self.id2rel[rel_label].get_symmetric:  # 即该关系三元组可对称
            head, tail = rel_triplet[0], rel_triplet[1]
            if tail[0] < head[0]:
                adjusted_rel = tail, head, rel_label
            return adjusted_rel

    def evaluation(self, pred_rels, true_rels):
        tp, fp, fn = {}, {}, {}
        each_class = {}

        pr = frozenset(pred_rels)
        tr = frozenset(true_rels)

        for rel_label in self.id2rel.keys():
            tp[rel_label] = 0
            fp[rel_label] = 0
            fn[rel_label] = 0

        for rel_l in self.id2rel.keys():
            tp[rel_l] += len([a for a in pr if a in tr and a[-1] == rel_l])
            fp[rel_l] += len([a for a in pr if a not in tr and a[-1] == rel_l])
            fn[rel_l] += len([a for a in tr if a not in pr and a[-1] == rel_l])
            each_class[rel_l] = self.prf(tp[rel_l], fp[rel_l], fn[rel_l])  # 每条关系的评估数据

        tp_total = np.sum([tp[a] for a in tp])  # in tp 同　in tp.keys(),所有类型tp的数量之和
        fp_total = np.sum([fp[a] for a in fp])  # fp_tot = len([a for a in pr if a not in tr])
        fn_total = np.sum([fn[a] for a in fn])  # fn_tot = len([a for a in tr if a not in pr])
        total_prf = self.prf(tp_total, fp_total, fn_total)  # 总体数据的评估, Micro

        mean_prf = np.array(each_class)
        mean_prf = np.mean(mean_prf, axis=0)  # 每条关系的评估平均值，Macro

        print("The evaluation result, micro: ", total_prf)
        print("The evaluation result, Macro: ", mean_prf)

    def prf(self, tp, fp, fn):
        p = float(tp) / (tp+fp) if (tp+fp != 0) else 0.0
        r = float(tp) / (tp+fn) if (tp+fn != 0) else 0.0
        f = ((2*p*r)/(p+r)) if p != 0.0 and r != 0.0 else 0.0
        return [p*100, r*100, f*100]
