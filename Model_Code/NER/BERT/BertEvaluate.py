#!/usr/bin/python3
# -*- coding: utf-8 -*-

import time
import datetime
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import sys
sys.path.append('..')
from data_utils import *

def time_format(time_diff):
    seconds = int(round(time_diff))
    return str(datetime.timedelta(seconds=seconds))

def bert_evaluate(eval_model, eval_data, label_alphabet, eval_epoch, batch_size, eval_device, eval_data_name):
    eval_model.eval()
    all_pred_labels = []
    all_true_labels = []
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for eval_batch in gen_batch(eval_data, batch_size):
            batch_data = eval_batch.to(eval_device)

            pred_label_ids = eval_model.decoder(batch_data['token'])
            print(pred_label_ids)
            all_pred_labels.extend(pred_label_ids)
            dev_pred_tensor = torch.tensor(pred_label_ids, dtype=torch.long).to(eval_device)

            dev_true_tensor = torch.tensor([label_alphabet.get_index(label) for label in batch_data['label']],
                                           dtype=torch.long)
            dev_true = dev_true_tensor.cpu().detach().tolist()
            print(dev_true)
            all_true_labels.extend(dev_true)

            assert len(all_true_labels) == len(all_pred_labels)

            total = total + len(dev_true)
            assert total == len(all_pred_labels)

            correct = correct + dev_pred_tensor.eq(dev_true_tensor).sum().item()

    assert len(all_true_labels) == len(all_pred_labels)
    average_acc = correct / total
    f1 = f1_score(np.array(all_pred_labels), np.array(all_true_labels), average='weighted')
    end = time.time()
    print("This is %s: \n Epoch: %d\n Acc: %.2f\n F1: %.2f\n Spending: %s"%
          (eval_data_name, eval_epoch, average_acc*100, f1*100, time_format(end-start)))
    return average_acc, f1



