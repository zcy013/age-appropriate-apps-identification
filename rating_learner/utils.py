#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# author: zcy

import json
import os
import random
import time
import numpy as np
import openpyxl

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, cross_val_score

from configs import dir_features_all, dir_features_img, random_state, market


def read_apk_features(path):
    apk_feature = {}
    try:
        with open(path, "rb") as handle:
            apk_feature = json.load(handle)
    except Exception as e:
        pass
        # print('error on reading %s: %s' % (path, e))
    return apk_feature


def get_all_feature_files_and_labels_checked(assign_market=None, file_group=''):
    if not assign_market:
        assign_market = market
    wb = openpyxl.load_workbook('../sampler/app_zip%s/apps_manual_check_%s.xlsx' % (file_group, file_group))
    ws = wb.worksheets[0]
    file_list = []
    labels = []
    for row in ws.iter_rows(min_row=2):
        if int(row[3].value) == 1 and int(row[4].value) != 1:
            pkg = row[0].value
            file_list.append(os.path.join(dir_features_all % assign_market, file_group, pkg + '.txt'))
            apk = read_apk_features(os.path.join(dir_features_all % assign_market, file_group, pkg + '.txt'))
            if apk['minAge'] < 12:
                labels.append(1)
            else:
                labels.append(0)
    return file_list, labels


def get_all_feature_files_and_labels(assign_market=None):
    global market
    if assign_market and assign_market != market:
        market = assign_market
    file_list = []
    labels = []
    for cls in ['kid', 'nkid']:
        for file in os.listdir(os.path.join(dir_features_all % (market, cls))):
            if file.endswith('.txt'):
                apk = read_apk_features(os.path.join(dir_features_all % (market, cls), file))
                if apk:
                    # if apk['minAge'] == -1 or apk['minAge'] == 12 or apk['minAge'] == 16:
                    #     continue
                    imgs = read_apk_features(os.path.join(dir_features_img % (market, cls), file))
                    if imgs:
                        file_list.append(os.path.join(dir_features_all % (market, cls), file))
                        if cls == 'kid':
                            labels.append(1)
                        else:
                            labels.append(0)
    return file_list, labels


def get_part_feature_files_and_labels(kid_count, adt_count, assign_market=None):
    global market
    if assign_market and assign_market != market:
        market = assign_market
    file_list = []
    labels = []
    kid_count_curr = 0
    adt_count_curr = 0

    for root, dirs, files in os.walk(os.path.join(dir_features_all_adt % market)):
        random.seed(random_state)
        random.shuffle(files)
        for f in files:
            if f.endswith('.txt'):
                apk = read_apk_features(os.path.join(dir_features_all_adt % market, f))
                if apk:
                    if apk['minAge'] == -1 or apk['minAge'] < 12:
                        continue
                    imgs = read_apk_features(os.path.join(dir_features_img_adt % market, f))
                    if imgs and adt_count_curr < adt_count:
                        adt_count_curr += 1
                        file_list.append(os.path.join(dir_features_all_adt % market, f))
                        labels.append(0)
    for root, dirs, files in os.walk(os.path.join(dir_features_all_kid % market)):
        random.seed(random_state)
        random.shuffle(files)
        for f in files:
            if f.endswith('.txt'):
                apk = read_apk_features(os.path.join(dir_features_all_kid % market, f))
                if apk:
                    if apk['minAge'] == -1 or apk['minAge'] >= 12:
                        continue
                    imgs = read_apk_features(os.path.join(dir_features_img_kid % market, f))
                    if imgs and kid_count_curr < kid_count:
                        kid_count_curr += 1
                        file_list.append(os.path.join(dir_features_all_kid % market, f))
                        labels.append(1)
    return file_list, labels


def libsvm_transformer(data_ndarray, labels, save_file):
    data_list = []
    total_file, total_col = data_ndarray.shape

    print(labels)
    with open(save_file, 'w') as file:
        for counter in range(total_file):
            file.write('%d' % labels[counter])
            data_dict = {}

            for index in range(total_col):
                data_dict[index+1] = data_ndarray[counter][index]
                file.write(' %d:%.6e' % (index+1, data_ndarray[counter][index]))
            file.write('\n')

            data_list.append(data_dict)
    return labels, data_list


def print_performance(labels_true, labels_pred):
    print('accuracy =', accuracy_score(labels_true, labels_pred))
    print('precision =', precision_score(labels_true, labels_pred))
    print('recall =', recall_score(labels_true, labels_pred))
    print('f1-score =', f1_score(labels_true, labels_pred))
    print('roc-auc-score =', roc_auc_score(labels_true, labels_pred))
    cm = confusion_matrix(labels_true, labels_pred)
    print('confusion matrix =', cm)
    # 顺序：   预测0（负）  预测1（正）
    # 实际0（负）
    # 实际1（正）
    tn, fp, fn, tp = cm.ravel()
    print('TPR =', tp / (tp + fn))
    print('TNR =', tn / (tn + fp))


def tnr_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)


tnr_scorer = make_scorer(tnr_score, greater_is_better=True)

"""
def train_models_grid(X, Y):
    kfold = StratifiedKFold(n_splits=10, random_state=random_state)
    # scorings = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    # refit = 'roc_auc'

    # scorings = {'prec': 'precision'}  #, 'auc': 'roc_auc'}  # , 'acc': 'accuracy'}
    # refit = 'prec'
    scorings = 'roc_auc'
    refit = True
    jobs = 1
    verbose = 0

    # TODO 太费时，单独跑
    print('[%s] Linear SVM:' % time.strftime('%Y-%m-%d %H:%M:%S'))
    model = SVC(random_state=random_state, probability=True)
    clf = GridSearchCV(model, [
        {'C': np.logspace(-5, 5, 11), 'gamma': np.logspace(-5, 5, 11), 'kernel': ['rbf']},
        {'kernel': ['linear'], 'C': np.logspace(-5, 5, 11)}
    ], scoring=scorings, refit=refit, cv=kfold, n_jobs=jobs, verbose=verbose)
    clf.fit(X, Y)
    print('best params:', clf.best_params_)
    print('best estimator:', clf.best_estimator_)
    print('best score:', clf.best_score_)
    prob = clf.predict_proba(X)
    pred = np.array([0, 1]).take(np.argmax(prob, axis=1), axis=0)
    print_performance(Y, pred)

    print('[%s] K-Nearest Neighbors:' % time.strftime('%Y-%m-%d %H:%M:%S'))
    model = KNeighborsClassifier()
    clf = GridSearchCV(model, [
        {'weights': ['uniform'], 'n_neighbors': np.linspace(1, 10, 10, dtype=int)},
        {'weights': ['distance'], 'n_neighbors': np.linspace(1, 10, 10, dtype=int), 'p': np.linspace(1, 5, 5, dtype=int)},
    ], scoring=scorings, refit=refit, cv=kfold, n_jobs=jobs, verbose=verbose)
    clf.fit(X, Y)
    print('best params:', clf.best_params_)
    print('best estimator:', clf.best_estimator_)
    print('best score:', clf.best_score_)
    prob = clf.predict_proba(X)
    pred = np.array([0, 1]).take(np.argmax(prob, axis=1), axis=0)
    print_performance(Y, pred)

"""
def train_models_grid(X, Y):
    kfold = StratifiedKFold(n_splits=10, random_state=random_state)
    # scorings = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    # refit = 'roc_auc'

    # scorings = {'prec': 'precision'}  #, 'auc': 'roc_auc'}  # , 'acc': 'accuracy'}
    # refit = 'prec'
    scorings = 'roc_auc'
    refit = True
    jobs = 1
    verbose = 0

    print('[%s] Decision Tree:' % time.strftime('%Y-%m-%d %H:%M:%S'))
    model = DecisionTreeClassifier(random_state=random_state)
    # kfold获得的是k个不同模型，best_score_是最优参数的k个模型的性能均值
    # 设置refit是为了在总的数据集上获得一个确定的模型
    clf = GridSearchCV(model, {
        'criterion': ['entropy', 'gini'], 'min_samples_leaf': np.linspace(1, 10, 10, dtype=int),
        'max_depth': [30, 50, 60, 100], 'min_impurity_decrease': [0.1, 0.2, 0.5]
    }, scoring=scorings, refit=refit, cv=kfold, n_jobs=jobs, verbose=verbose)
    clf.fit(X, Y)
    print('best params:', clf.best_params_)
    print('best estimator:', clf.best_estimator_)
    # 这个值用来评估参数和模型在训练集上的性能
    print('best score:', clf.best_score_)  # Mean cross-validated score of the best_estimator.
    # print(clf.cv_results_)
    prob = clf.predict_proba(X)  # 等同于clf.best_estimator.predict_proba(X)
    pred = np.array([0, 1]).take(np.argmax(prob, axis=1), axis=0)
    # 如果前面设置了refit，这里就是在训练集上训练又在训练集上测试的结果，没有意义
    # 但之后要组合模型，需要确定的单个模型，所以仅用来与组合模型的性能做比较（因为组合模型不再做kfold）
    print_performance(Y, pred)

    print('[%s] Logistic Regression:' % time.strftime('%Y-%m-%d %H:%M:%S'))
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    clf = GridSearchCV(model, {
        'C': np.logspace(-5, 5, 11), 'solver': ['liblinear', 'lbfgs', 'newton-cg', 'saga'],
    }, scoring=scorings, refit=refit, cv=kfold, n_jobs=jobs, verbose=verbose)
    clf.fit(X, Y)
    print('best params:', clf.best_params_)
    print('best estimator:', clf.best_estimator_)
    print('best score:', clf.best_score_)
    prob = clf.predict_proba(X)
    pred = np.array([0, 1]).take(np.argmax(prob, axis=1), axis=0)
    print_performance(Y, pred)

    print('[%s] Random Forest:' % time.strftime('%Y-%m-%d %H:%M:%S'))
    model = RandomForestClassifier(random_state=random_state)
    clf = GridSearchCV(model, {
        # 'max_features': [50],
        'criterion': ['entropy', 'gini'], 'min_samples_leaf': np.linspace(1, 10, 10, dtype=int),
        'n_estimators': [10, 30, 50, 100, 150, 200, 300]
    }, scoring=scorings, refit=refit, cv=kfold, n_jobs=jobs, verbose=verbose)
    clf.fit(X, Y)
    print('best params:', clf.best_params_)
    print('best estimator:', clf.best_estimator_)
    print('best score:', clf.best_score_)
    prob = clf.predict_proba(X)
    pred = np.array([0, 1]).take(np.argmax(prob, axis=1), axis=0)
    print_performance(Y, pred)

    print('[%s] Multi-Layer Perceptron:' % time.strftime('%Y-%m-%d %H:%M:%S'))
    model = MLPClassifier(random_state=random_state)
    clf = GridSearchCV(model, {
        'hidden_layer_sizes': [(25,), (50,), (75,), (100,), (125,), (150,), (200,)],
        'max_iter': [20, 50, 100, 150, 200, 250, 300], 'solver': ['adam', 'sgd', 'lbfgs']
    }, scoring=scorings, refit=refit, cv=kfold, n_jobs=jobs, verbose=verbose)
    clf.fit(X, Y)
    print('best params:', clf.best_params_)
    print('best estimator:', clf.best_estimator_)
    print('best score:', clf.best_score_)
    prob = clf.predict_proba(X)
    pred = np.array([0, 1]).take(np.argmax(prob, axis=1), axis=0)
    print_performance(Y, pred)


def train_default_models(X, Y):
    kfold = StratifiedKFold(n_splits=10, random_state=random_state)
    scorings = {'acc': 'accuracy', 'prec': 'precision', 'recall': 'recall', 'tnr': tnr_scorer, 'f1': 'f1', 'auc': 'roc_auc'}

    models = [DecisionTreeClassifier(), LogisticRegression(), SVC(), BernoulliNB(), RandomForestClassifier(), MLPClassifier()]
    for model in models:
        print('\n[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), model))
        # scores = cross_val_score(model, X, Y, cv=kfold)
        scores = cross_validate(model, X, Y, cv=kfold, scoring=scorings, n_jobs=-1, return_train_score=True)
        for k, v in scores.items():
            print('%s avg = %f' % (k, np.mean(v)))
            print(v)
