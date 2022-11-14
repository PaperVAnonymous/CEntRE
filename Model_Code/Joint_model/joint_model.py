#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import sys
# rf https://scikit-learn.org.cn/view/497.html
from sklearn.metrics import precision_recall_fscore_support as prfs  # =precision_score+recall_score+f1_score
sys.path.append('..')
from Joint_model.joint_data import JointData
from NER.ner_model import NER_module
from RelE.re_model import RE_module

class JointModule(nn.Module):
    def __init__(self, config):
        super(JointModule, self).__init__()
        self.conf = config
        self.joint_data = JointData(self.conf)  # joint_data对象
        self.ner_data = self.joint_data.get_nerdata_object  # ner_data对象

        self.train_data_ids = self.ner_data.train_ids
        self.dev_data_ids = self.ner_data.dev_ids
        self.test_data_ids = self.ner_data.test_ids
        self.label_alphabet = self.ner_data.label_alphabet

        self.ner_model = NER_module(self.conf, self.ner_data)  # ner模型
        self.re_model = RE_module(self.conf, self.label_alphabet)  # re模型
        self.index2rel = self.re_model.id2rel
        self.ner_labels = self.re_model.index_OBI

        self.all_ner_preds = []
        self.all_ner_trues = []

        self.all_rel_preds = []
        self.all_rel_trues = []

    def joint_train(self, data_index, mode):
        if mode == 'train':
            instance_ids = self.train_data_ids[data_index]
        elif mode is 'valid':
            instance_ids = self.dev_data_ids[data_index]

        ner_object_score = self.ner_model.ner_train(instance_ids)

        hidden_emb = self.ner_model.fused_gate(instance_ids)
        re_object_score = self.re_model.re_train(instance_ids[-1], hidden_emb)  #

        return ner_object_score+re_object_score

    def joint_test(self, data_index):
        instance_ids = self.test_data_ids[data_index]
        test_ner_pred, test_ner_true = self.ner_model.ner_test(instance_ids)
        assert len(test_ner_pred) == len(test_ner_true)  # 必须保证预测的序列长度和真实的序列标签长度一致
        self.all_ner_preds.append(test_ner_pred)  # 便于对整体的测试数据进行评估
        self.all_ner_trues.append(test_ner_true)  # list[list]

        hidden_emb = self.ner_model.fused_gate(instance_ids)
        test_rel_pred, test_rel_true = self.re_model.re_test(instance_ids[-1], hidden_emb, test_ner_pred)
        self.all_rel_preds.append(test_rel_pred)
        self.all_rel_trues.append(test_rel_true)  # list[list[tuple]]


    def ner_eval(self):
        pr_flat = []
        tr_flat = []
        for pred_sent, true_sent in zip(self.all_ner_preds, self.all_ner_trues):
            pr_flat+=pred_sent
            tr_flat+=true_sent

        weight = prfs(tr_flat, pr_flat, labels=self.ner_labels, average='weighted', zero_division='warn')[:-1]
        print("NER evaluation, weight: %s" % [each_ele * 100 for each_ele in weight])
        per_type = prfs(tr_flat, pr_flat, labels=self.ner_labels, average=None, zero_division='warn')[:-1]
        print("NER evaluation, per_type: %s" % [each_ele*100 for each_ele in per_type])
        micro = prfs(tr_flat, pr_flat, labels=self.ner_labels, average='micro', zero_division='warn')[:-1]
        print("NER evaluation, micro: %s", [each_ele * 100 for each_ele in micro])
        macro = prfs(tr_flat, pr_flat, labels=self.ner_labels, average='macro', zero_division='warn')[:-1]
        print("NER evaluation, macro: %s", [each_ele * 100 for each_ele in macro])


    def rel_eval(self):
        tr_flat = []
        pr_flat = []
        rel_labels = set()  # 获取只存在与测试数据中的，pred和true中出现的标签信息

        for (pred_rels, true_rels) in zip(self.all_rel_preds, self.all_rel_trues):
            # 处理每一条数据
            union = set()
            union.update(pred_rels)
            union.update(true_rels)  # 当前union中包含了pred_rels和true_rels的并集

            for each_rel in union:

                if each_rel in pred_rels:
                    pr_flat.append(each_rel[-1])  # 添加关系标签
                    rel_labels.add(each_rel[-1])
                else:
                    pr_flat.append(-1)  # 针对true_rels的那个部分进行填充

                if each_rel in true_rels:
                    tr_flat.append(each_rel[-1])
                    rel_labels.add(each_rel[-1])
                else:
                    tr_flat.append(-1)  # 针对pred_rels的那个部分进行填充

        self.compute_metrics(tr_flat, pr_flat, list(rel_labels))

        # metrics = self.compute_metrics(tr_flat, pr_flat, rel_labels)
        # return metrics

    def compute_metrics(self, tr, pr, rel_labels):
        weight = prfs(tr, pr, labels=rel_labels, average='weighted', zero_division='warn')[:-1]
        per_type = prfs(tr, pr, labels=rel_labels, average=None, zero_division='warn')
        micro = prfs(tr, pr, labels=rel_labels, average='micro', zero_division='warn')[:-1]
        macro = prfs(tr, pr, labels=rel_labels, average='macro', zero_division='warn')[:-1]
        total_support = sum(per_type[-1])  # true数据的数据量

        print("The relation extraction evaluation of weight: precision, recall, f1_score\n")
        print([each_ele*100 for each_ele in weight])
        print("The relation extraction evaluation of per_type: precision, recall, f1_score\n")
        print([each_ele*100 for each_ele in per_type[:-1]])
        print("The relation extraction evaluation of micro: precision, recall, f1_score\n")
        print([each_ele*100 for each_ele in micro])
        print("The relation extraction evaluation of macro: precision, recall, f1_score\n")
        print([each_ele*100 for each_ele in macro])

        print("The relation extraction evaluation of precision, recall, f1_score, support for each type!\n")

        self.print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], rel_labels)


    def print_results(self, per_type, micro, macro, types):
        columns = ('type', "precision", "recall", 'f1_score', 'support')

        # 输出格式设置
        row_fmt = '%20s' + ("%12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):  # i表示是第几个类
            metrics = []
            for j in range(len(per_type)):  # j表示的是precision/recall/f1_score/support
                metrics.append(per_type[j][i])  # metrics表示每个类的precision/recall/f1_score/support信息
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self.get_row(m, self.index2rel[t].get_name))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self.get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self.get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)


    def get_row(self, eva_result, eva_name):
        """
        :param eva_result: a list containing 4 elements
        :param eva_name: a string name
        :return: a tuple containing 4 elements
        """
        row = [eva_name]
        for i in range(len(eva_result) - 1):  # -1说明最后的support是没有添加到输出的
            row.append('%.2f' % (eva_result[i] * 100))
        row.append(eva_result[-1])

        return tuple(row)  # 将列表内容转换成为元组，方便进行输出
