#!/usr/bin/python3
# -*- coding: utf-8 -*-



import os
import torch
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange
import torch.optim as optim
from config import BasicArgs as args
from data_utils import *
import sys
sys.path.append('..')
from Joint_model.joint_model import JointModule as joint_model
from Joint_model.L2_reg import L2

if __name__ == "__main__":
    if not os.path.exists(args.JointModel_saved_path):
        os.makedirs(args.JointModel_saved_path)

    model = joint_model(args).to(args.device)
    l2_reg = L2(model, 0.5)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_decay_factor,
                                                    verbose=True, patience=3, min_lr=args.min_lr)
    train_num = len(args.train_data)
    valid_num = len(args.dev_data)
    test_num = len(args.test_data)

    train_epoch_loss = []
    valid_epoch_loss = []

    for epoch in trange(args.total_epoch, desc='Epoch'):
        train_loss = 0
        train_start = time.time()
        batch_start = time.time()

        model.train()
        model.zero_grad()
        for train_iter, instance_index in enumerate(random_index(train_num, 1)): # 获取train_data的每次索引
            instance_loss = model.joint_train(instance_index, 'train')
            l2_instance_loss = instance_loss+l2_reg(model)
            l2_instance_loss.backward()
            train_loss = train_loss + l2_instance_loss.cpu().item()
            optimizer.step()
            optimizer.zero_grad()

            if train_iter % 20 == 0 and train_iter != 0:
                print("20 instances cost time: {}".format(time.time()-batch_start))
                print("Epoch:{}-{}/{}, Loss:{}".format(epoch, train_iter, train_num, l2_instance_loss))
                batch_start = time.time()

        train_ave_loss = train_loss/train_num  # 求取本村epoch里面所有数据的平均loss
        train_epoch_loss.append(train_ave_loss)
        print('Epoch: {} is completed, and the average train loss is: {}, spend: {}'.format(epoch, train_ave_loss, time.time()-train_start))
        print("**************************Let us begin the validation of epoch {}****************************".format(epoch))

        # Then, we will evaluate and save the model
        model.eval()
        valid_loss = 0
        for valid_iter, valid_instance_index in enumerate(random_index(valid_num, 1)):
            valid_instance_loss = model.joint_train(valid_instance_index, 'valid')
            l2_valid_instance_loss = valid_instance_loss + l2_reg(model)
            valid_loss = valid_loss + l2_valid_instance_loss.detach().cpu().item()
        valid_ave_loss = valid_loss/valid_num
        valid_epoch_loss.append(valid_ave_loss)
        print("Validation: Epoch-{}, Val_loss-{}".format(epoch, valid_ave_loss))

        lr_decay.step(valid_ave_loss)

    torch.save({'model_state': model.state_dict()}, os.path.join(args.JointModel_saved_path, 'model.checkpoint.pt'))

    print("**********************************************\n"
          "********     The training is over.    ********\n"
          "**********************************************")

    test_checkpoint = torch.load(os.path.join(args.JointModel_saved_path, 'model.checkpoint.pt'), map_location='cpu')
    model_dict = test_checkpoint['model_state']
    test_model_dict = model.state_dict()
    selected_model_state = {k: v for k, v in model_dict.items() if k in test_model_dict}
    test_model_dict.update(selected_model_state)
    model.load_state_dict(test_model_dict)
    model.to(args.device)
    model.eval()
    for test_iter, test_instance_index in enumerate(random_index(test_num, 1)):
        model.joint_test(test_instance_index)  # 对所有数据进行预测
    model.ner_eval()  # 进行NER评估
    model.rel_eval()  # 进行relation评估


    # rf https://blog.csdn.net/qq_44315987/article/details/104047632
    # sns.despine()
    sns.set(style="ticks", font='cmr10', font_scale=1.5)  # 设置背景，设置字体大小; 同样于 sns.set_context(context="paper", font_scale=1.5)
    plt.figure(figsize=(12,6))  # 同样于 plt.rcParams['figure.figsize'] = [12,6]

    # color rf https://blog.csdn.net/jirryzhang/article/details/77374702
    plt.plot(train_epoch_loss, color='b', label='TrainLoss', linewidth=1.5) # blue
    plt.plot(valid_epoch_loss, color='r', label='ValidLoss', linewidth=1.5) # red

    plt.title('Learning Curve')
    plt.xlabel('Epoch')  # 横轴说明
    plt.ylabel('Loss')  # 纵轴说明
    plt.legend(['TrainLoss', 'ValidLoss'], loc='upper right')
    # plt.legend(['train_loss', 'valid_loss'])
    plt.savefig('model.jpg')
    plt.show()
