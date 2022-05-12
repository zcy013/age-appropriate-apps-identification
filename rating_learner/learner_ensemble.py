#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# author: zcy

import os
import time
import numpy as np
import itertools
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier


import learner_basic,  learner_text, learner_image, learner_text_tfidf
from configs import market, model2features, model_weights, MODEL_TEXT, MODEL_IMGS, MODEL_OCRS, MODEL_CATE, MODEL_ADS, MODEL_HSV, MODEL_NAME, MODEL_RATE, MODEL_CODE, random_state
from utils import get_all_feature_files_and_labels, get_part_feature_files_and_labels, get_all_feature_files_and_labels_checked, print_performance


file_model = '%s/model_%s.pkl'
file_vals = '%s/vals_%s.json'


class MyDataset:
    def __init__(self, files=None, labels=None, kid_count=0, adt_count=0, assign_market=None, checked=False):
        if files:
            self.files = files
            self.labels = labels
        elif checked:
            self.files, self.labels = [], []
            for fileg in file_group:
                files, labels = get_all_feature_files_and_labels_checked(assign_market, str(fileg))
                self.files += files
                self.labels += labels
        else:
            if kid_count + adt_count == 0:
                self.files, self.labels = get_all_feature_files_and_labels(assign_market)
            else:
                self.files, self.labels = get_part_feature_files_and_labels(kid_count, adt_count, assign_market)
        print(len(self.files), len(self.labels))
        self.files_train = []
        self.files_test = []
        self.labels_train = []
        self.labels_test = []
        self.files_train, self.files_test, self.labels_train, self.labels_test = train_test_split(self.files, self.labels, random_state=random_state)
        if self.labels:
            self.pos_count = self.labels.count(1)
            self.neg_count = len(self.labels) - self.pos_count


def train_save_model(market, name, files, labels):
    val = None
    if name == MODEL_NAME:
        learner = learner_text_tfidf.get_trained_model(name, files, labels)
    elif name == MODEL_TEXT:
        learner = learner_text.get_trained_model(files, labels)
    elif name == MODEL_IMGS:
        learner, dict = learner_image.get_trained_model(name, files, labels)
        val = [dict]
    elif name == MODEL_HSV:
        learner = learner_image.get_trained_model(name, files, labels)
    elif name == MODEL_OCRS:
        learner = learner_image.get_trained_model(name, files, labels)
    elif name == MODEL_ADS or name == MODEL_RATE:
        learner = learner_basic.get_trained_model(name, files, labels, market)
    else:
        learner, dict, keys = learner_basic.get_trained_model(name, files, labels, market)
        val = [dict, keys]

    joblib.dump(learner, file_model % (market, name))
    if val:
        with open(file_vals % (market, name), "w", encoding='utf-8') as handle:
            json.dump(val, handle, ensure_ascii=False)
    return learner, val


def load_model(market, name):
    model = joblib.load(file_model % (market, name))
    if os.path.exists(file_vals % (market, name)):
        with open(file_vals % (market, name), "rb") as handle:
            val = json.load(handle)
            return model, val
    return model, None


def model_predict_proba(name, files, learner, val, assign_market):
    if name == MODEL_NAME:
        return learner_text_tfidf.predict_proba(name, files, learner)
    elif name == MODEL_TEXT:
        return learner_text.predict_proba(files, learner)
    elif name == MODEL_IMGS:
        return learner_image.predict_proba(name, files, learner, val)
    elif name == MODEL_OCRS:
        return learner_image.predict_proba(name, files, learner)
    elif name == MODEL_HSV:
        return learner_image.predict_proba(name, files, learner)
    else:
        return learner_basic.predict_proba(name, files, learner, val, assign_market)


# 在75%训练集上训练模型并保存
def train_models(dataset=None):
    print('\n[%s] build dataset ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    if not dataset:
        dataset = MyDataset()
        dataset = MyDataset(files=dataset.files_train, labels=dataset.labels_train)

    print('\n\n[%s] training models ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    models = {}
    vals = {}
    # TODO multi-process
    for name in model2features.keys():
        # if not os.path.exists(file_model % (market, name)):
        models[name], vals[name] = train_save_model(market, name, dataset.files_train, dataset.labels_train)
        print('\n[%s] got %s model' % (time.strftime('%Y-%m-%d %H:%M:%S'), name))
    return dataset, models, vals


# 在25%训练集上将训练好的模型尝试所有组合方式
def combine_models(dataset=None, models=None, vals=None, assign_market=None):
    print('\n\n[%s] ensemble models' % time.strftime('%Y-%m-%d %H:%M:%S'))
    if not dataset:
        print('[%s] build dataset ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
        dataset = MyDataset(assign_market=assign_market)
        dataset = MyDataset(files=dataset.files_train, labels=dataset.labels_train)
    if not models or not vals:
        models = {}
        vals = {}
        for name in model2features.keys():
            models[name], vals[name] = load_model(market, name)
            print('[%s] got %s model' % (time.strftime('%Y-%m-%d %H:%M:%S'), name))

    # 文本特征的向量化很花时间，所以将每个模型的预测标签保存，避免重复计算
    prob = {}
    preds = []
    files = dataset.files_test
    labels = dataset.labels_test
    print(np.array(labels))
    for name in model2features.keys():
        prob[name] = model_predict_proba(name, files, models[name], vals[name], assign_market=assign_market if assign_market else market)
        pred = np.array([0, 1]).take(np.argmax(prob[name], axis=1), axis=0)  # 正负类的预测（概率较大的那个）
        print(pred)
        preds.append(pred)
        print('\n[%s] model: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), name))
        print_performance(labels, pred)
    print(','.join(['pkg', 'label'] + list(model2features.keys())))
    for i in range(len(labels)):
        print(','.join([files[i].split('/')[-1], str(labels[i])] + [str(pred[i]) for pred in preds]))

    # 所有可能的模型组合
    # 单个模型的数量
    for num_models in range(2, len(models) + 1):
        models_part = list(itertools.combinations(models.items(), num_models))
        # 一种组合方式
        for com in models_part:
            print('\n[%s] models: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), ', '.join([m[0] for m in com])))
            prob_sum = np.zeros((len(labels), 2))  # 正负类的概率
            # weight_sum = 0
            # 累加组合中的每个模型
            for name, model in com:
                # print(prob[name])
                prob_sum += model_weights[name] * prob[name]  # 所有模型的预测概率的和
                # weight_sum += model_weights[name]  # 不需要除以总weight，因为后续只用到了正负例预测得分的大小关系，数值的scale不重要
            # print(prob_sum)
            # print('\navg proba (kid):\n', probs[:, 1] / len(models.keys()))
            predictions = np.array([0, 1]).take(np.argmax(prob_sum, axis=1), axis=0)
            print(predictions)
            # print('prediction: \n', list(predictions))
            # print('ground truth: \n', dataset.labels_test)
            print_performance(labels, predictions)


# 跨市场应用模型
def cross_markets(markets):
    for src_market in markets:
        print('\n\n[%s] training dataset: %s ...' % (time.strftime('%Y-%m-%d %H:%M:%S'), src_market))
        models = {}
        vals = {}
        for name in model2features.keys():
            if os.path.exists(file_model % (src_market, name)):
                models[name], vals[name] = load_model(src_market, name)
            else:
                # dataset = MyDataset(assign_market=src_market)
                dataset = MyDataset()
                print(dataset.pos_count, dataset.neg_count)
                models[name], vals[name] = train_save_model(src_market, name, dataset)
            print('[%s] got %s model' % (time.strftime('%Y-%m-%d %H:%M:%S'), name))
            # print(models[name], vals[name])

        for tar_market in markets:
            print('\n\n[%s] testing dataset: %s ...' % (time.strftime('%Y-%m-%d %H:%M:%S'), tar_market))
            # dataset2 = MyDataset(assign_market=tar_market, file_group='13', checked=True)
            dataset2 = MyDataset(assign_market=tar_market)
            print(dataset2.labels_test.count(1), dataset2.labels_test.count(0))
            prob_sum = np.zeros((len(dataset2.labels_test), 2))
            preds = []
            for name in model2features.keys():
                prob = model_predict_proba(name, dataset2.files_test, models[name], vals[name], tar_market)
                pred = np.array([0, 1]).take(np.argmax(prob, axis=1), axis=0)  # 正负类的预测（概率较大的那个）
                preds.append(pred)
                print('\n[%s] model: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), name))
                print_performance(dataset2.labels_test, pred)
                # prob_sum += prob
                prob_sum += model_weights[name] * prob  # 所有模型的预测概率的和
            predictions = np.array([0, 1]).take(np.argmax(prob_sum, axis=1), axis=0)
            print('\n[%s] ensemble:' % (time.strftime('%Y-%m-%d %H:%M:%S')))
            print_performance(dataset2.labels_test, predictions)
            print('\t'.join(['pkg', 'label', 'prediction', 'correct'] + list(model2features.keys())))
            for i in range(len(dataset2.labels_test)):
                print('\t'.join([dataset2.files_test[i], str(dataset2.labels_test[i]), str(predictions[i]),
                                str(dataset2.labels_test[i] == predictions[i])] + [str(pred[i]) for pred in preds]))
        break


def train_all(dataset=None, grid=False):
    # 注意！这三行要注意是否要一起改！！
    # if not dataset:
    #     dataset = MyDataset(file_group=file_group)
    files = dataset.files_train
    labels = dataset.labels_train
    for name in model2features.keys():
        print('\n\n\n\nfeature:', name)
        if name == MODEL_TEXT:
            learner_text.train_callee(files, labels, grid)
        elif name == MODEL_IMGS:
            learner_image.train_callee(name, files, labels, grid)
        elif name == MODEL_OCRS:
            learner_image.train_callee(name, files, labels, grid)
        else:
            learner_basic.train_callee(name, files, labels, grid)


# 调用集成模型并解释
def explain_models(file, label):
    for name in model2features.keys():
        model, val = load_model(market, name)
        print('model:', name)
        if name == MODEL_TEXT:
            learner_text.explain_model(model, file, label)
        elif name == MODEL_OCRS:
            learner_image.explain_model(name, model, file, label)
        elif name == MODEL_CODE:
            learner_basic.explain_model(name, model, val, file, label)


if __name__ == '__main__':
    # model, val = train_save_model('huawei', 'image', MyDataset())
    # # model, val = load_model('huawei', 'code')

    # dataset = MyDataset()

    # print(np.array(dataset.labels_train))
    # prob = model_predict_proba('image', dataset.files_train, model, val, 'huawei')
    # pred = np.array([0, 1]).take(np.argmax(prob, axis=1), axis=0)  # 正负类的预测（概率较大的那个）
    # print(pred)
    # print('\n[%s] train: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), 'image'))
    # print_performance(dataset.labels_train, pred)
    #
    # print(np.array(dataset.labels_test))
    # prob = model_predict_proba('image', dataset.files_test, model, val, 'huawei')
    # pred = np.array([0, 1]).take(np.argmax(prob, axis=1), axis=0)  # 正负类的预测（概率较大的那个）
    # print(pred)
    # print('\n[%s] test: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), 'image'))
    # print_performance(dataset.labels_test, pred)

    # train_save_model('google', 'category', dataset.files_train, dataset.labels_train)
    # train_save_model('google', 'name', dataset.files_train, dataset.labels_train)
    # train_models()  # 用train
    # combine_models()
    # combine_models(assign_market='huawei')  # 用val（train.test）

    cross_markets(['google', 'huawei'])
    # dataset = MyDataset(kid_count=5, adt_count=5)
    # for file, label in zip(dataset.files, dataset.labels):
    #     explain_models(file, label)
    # train_all(dataset, grid=False)  # huawei 3 best

    # cases = {
    #     1: ['com.sinyee.babybus.meet.txt', 'com.viva.kidsflix.txt', 'com.tocaboca.tocalifeworld.txt'],
    #     0: ['com.craftsman.go.txt', 'com.lajeuneandassociatesllc.barberchopdev.txt',
    #         'com.hdictionary.bn.txt', 'photo.smile.red.eyes.changer.color.txt',
    #         'net.jeu.spider.txt', 'com.dumpgames.farm.animal.simulator.family.farming.txt']}
    # for label, files in cases.items():
    #     for file in files:
    #         path = '../data/google/%s/features_all/%s' % (['nkid', 'kid'][label], file)
    #         print('\n' + path)
    #         explain_models(path, label)
