#!/usr/bin/python3
# encoding: utf-8

import os
import operator
import re
import time
import pandas as pd
import numpy as np
import collections

from sklearn.model_selection import cross_val_score, train_test_split  # 交叉验证
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE, SelectKBest, SelectPercentile, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from lime.lime_tabular import LimeTabularExplainer

from utils import read_apk_features, get_all_feature_files_and_labels, get_part_feature_files_and_labels, libsvm_transformer, train_models_grid, train_default_models
from configs import features_to_count, model2features, market
from configs import MODEL_MANI, MODEL_CODE, MODEL_CATE, MODEL_ADS, MODEL_RATE
from configs import category_mappings, feature_size_assigned, random_state

apis2permissions_files = ['./Mappings-annotations.txt', './Mappings-docs.txt']
dangerous_api_union = '../api_union_23_30.txt'
dangerous_per_union = '../permission_union_23_30_dan.txt'
signature_per_union = '../permission_union_23_30_sig.txt'
normal_per_union = '../permission_union_23_30_nor.txt'
permission_type = {}


# 统计data中每个子数据的出现次数，累计到counter
# 已知api在统计时有bug，有重复计数，故在此处去重
# # todo tmp
# def count_sub_data(counter, data, feature_name):
#     global permission_type
#     if feature_name == 'permissions':  # 只保留危险权限
#         signature = []
#         dangerous = []
#         normal = []
#         with open(signature_per_union, 'r') as f:
#             for line in f.readlines():
#                 signature.append(line.strip())
#         with open(dangerous_per_union, 'r') as f:
#             for line in f.readlines():
#                 dangerous.append(line.strip())
#         with open(normal_per_union, 'r') as f:
#             for line in f.readlines():
#                 normal.append(line.strip())
#         for sub, detail in data[feature_name].items():
#             type_ = ''
#             if sub in signature:
#                 type_ += 'S'
#             if sub in dangerous:
#                 type_ += 'D'
#             if sub in normal:
#                 type_ += 'N'
#             if 'Unknown' in detail[1]:
#                 type_ += 'T'
#             if not type_:
#                 type_ += 'U'
#             # if sub in dangerous:
#             # if 'dangerous' in detail[0]:  # or 'signature' in detail[0]:
#             if sub in counter:
#                 counter[str(sub)] += 1
#             else:
#                 permission_type[sub] = type_
#                 counter[str(sub)] = 1
#     else:
#         for sub in set(data[feature_name]):
#             if sub in counter:
#                 counter[str(sub)] += 1
#             else:
#                 counter[str(sub)] = 1

# 统计data中每个子数据的出现次数，累计到counter
# 已知api在统计时有bug，有重复计数，故在此处去重
def count_sub_data(counter, data, feature_name):
    if feature_name == 'permissions':  # 只保留危险权限
        # dangerous = []
        # with open(dangerous_per_union, 'r') as f:
        #     for line in f.readlines():
        #         dangerous.append(line.strip())
        for sub, detail in data[feature_name].items():
            # if sub in dangerous:
            # if 'dangerous' in detail[0]:  # or 'signature' in detail[0]:
            if sub in counter:
                counter[str(sub)] += 1
            else:
                counter[str(sub)] = 1
    else:
        for sub in set(data[feature_name]):
            if sub in counter:
                counter[str(sub)] += 1
            else:
                counter[str(sub)] = 1


# 根据总数量，取前N%
def top_data_filter(dicts, total_count, filter_ratio):
    print('before filtering:', len(dicts.items()))
    print(total_count, filter_ratio, total_count * filter_ratio)
    sort_dict = dict(sorted(dicts.items(), key=operator.itemgetter(1), reverse=True))
    sort_dict = dict((key, value) for key, value in sort_dict.items() if value > total_count * filter_ratio)
    # for k, v in sort_dict.items():
    #     print(v, k)
    print('after filtering:', len(sort_dict.items()))
    return list(sort_dict.keys())


# 根据总数量，取前N%
def top_data_filter2(dicts, total_count, filter_ratio):
    print('before filtering:', len(dicts.items()))
    print(total_count, filter_ratio, total_count * filter_ratio)
    sort_dict = dict(sorted(dicts.items(), key=operator.itemgetter(1), reverse=True))
    sort_dict = dict((key, value) for key, value in sort_dict.items() if value < total_count * filter_ratio)
    # for k, v in sort_dict.items():
    #     print(v, k)
    print('after filtering:', len(sort_dict.items()))
    return list(sort_dict.keys())


# 统计apk的指定feature的指定子feature
def sub_features_encoder(feature_map, feature_name, apk):
    sub = []
    for f in feature_map[feature_name]:
        if isinstance(apk[feature_name], collections.Iterable):
            status = 1 if f in apk[feature_name] else 0
        else:
            status = 1 if f == apk[feature_name] else 0
        sub.append(status)
    return sub


def category_encoder(feature_map, feature_name, apk):
    category = apk[feature_name]  # .split('-')[0]
    cate = []
    for k, v in feature_map[feature_name].items():
        status = 1 if category in v else 0
        cate.append(status)
    return cate


def extract_string_feature(string_feature):
    pattern = re.compile(r'<string.*>(.*)</string>')
    matched = re.findall(pattern, string_feature)
    return ' '.join(matched)


# 这是我手动删除一些通用的api,这个你也可以删除
def delete_apis_manually(feature_list):
    useless_apis = ['java/lang', 'java/util',
                    'android/text', 'android/os/Looper', 'android/util', 'android/widget/',
                    'org/json', 'printStackTrace', '<init>']
    new_feature_list = []
    contains = False
    for A in feature_list:
        for api in useless_apis:
            if api in A:
                contains = True
        if not contains:
            new_feature_list.append(A)
        contains = False
    return new_feature_list


# 过滤api，也过滤应用（不考虑不包含任何重点api的应用）
def select_mapped_apis(feature_list, dataset):
    # 过滤api
    new_feature_list = []
    for feature in feature_list:
        matched = False
        feature_tmp = feature.replace('/', '.').replace(';->', '.')
        # for file in apis2permissions_files:
        with open(dangerous_api_union, 'r') as f:
            lines = f.readlines()
            for line in lines:
                mapped_api = line[: line.find('(')]
                mapped_api = 'L%s()' % mapped_api
                if feature_tmp == mapped_api:
                    new_feature_list.append(feature)
                    # matched = True
                    break
        # 修改上面时注意这段的缩进
        # if matched:
        #     break
    # 过滤应用 # TODO 暂时添加的功能
    # new_file_list = []
    # new_labels = []
    # for file, label in zip(dataset.file_list, dataset.labels):
    #     keep_this_file = False
    #     apk = read_apk_features(file)
    #     for feature_name in dataset.feature_list:
    #         features = apk[feature_name]
    #         for feature in new_feature_list:
    #             if feature in features:
    #                 keep_this_file = True
    #                 break
    #     if keep_this_file:
    #         new_file_list.append(file)
    #         new_labels.append(label)
    # dataset.file_list = new_file_list
    # dataset.labels = new_labels
    # dataset.pos_count = dataset.labels.count(1)
    # dataset.neg_count = len(dataset.labels) - dataset.pos_count
    return new_feature_list


def select_best_chi2(X, Y, ratio, size=0):
    global permission_type
    if size:
        model = SelectKBest(chi2, k=size)
    else:
        model = SelectPercentile(chi2, percentile=ratio)
    X_new = model.fit_transform(X, Y)
    # print(model.scores_)  # 得分越高，特征越重要
    # print(model.pvalues_)  # p-values 越小，置信度越高，特征越重要
    # mask = model.get_support()  # 得到每个特征是否被选择的T-F数组
    # indices = model.get_support(indices=True)  # 得到所有选择出的特征的索引数组（按索引顺序）
    indices = np.argsort(model.scores_)[::-1]  # 得到所有选择出的特征的索引数组（按score倒序）
    # best_features = list(X.columns.values[indices[0:feature_size]])
    for i in range(X_new.shape[1]):
        index = indices[i]
        if model.scores_[index] == 0:
            break
        print('%d\t%s\t%f\t%f' % (i+1, X.columns.values[index], model.scores_[index], model.pvalues_[index]))
        # print('%d\t%s\t%s\t%f' % (i+1, X.columns.values[index], permission_type[X.columns.values[index]], model.scores_[index]))
    X_new = pd.DataFrame(X_new, columns=X.columns.values[model.get_support(indices=True)])
    return X_new


def train_models(X, Y):
    print('[%s] Decision Tree:' % time.strftime('%Y-%m-%d %H:%M:%S'))
    model = DecisionTreeClassifier()
    scores = cross_val_score(model, X, Y, cv=10)
    print(scores)
    print('avg = %f' % np.mean(scores))

    print('[%s] Logistic Regression:' % time.strftime('%Y-%m-%d %H:%M:%S'))
    model = LogisticRegression()
    scores = cross_val_score(model, X, Y, cv=10)
    print(scores)
    print('avg = %f' % np.mean(scores))

    print('[%s] Linear SVM:' % time.strftime('%Y-%m-%d %H:%M:%S'))
    model = SVC()
    scores = cross_val_score(model, X, Y, cv=10)
    print(scores)
    print('avg = %f' % np.mean(scores))

    print('[%s] K-Nearest Neighbors:' % time.strftime('%Y-%m-%d %H:%M:%S'))
    model = KNeighborsClassifier()
    scores = cross_val_score(model, X, Y, cv=10)
    print(scores)
    print('avg = %f' % np.mean(scores))

    print('[%s] Random Forest:' % time.strftime('%Y-%m-%d %H:%M:%S'))
    model = RandomForestClassifier()
    scores = cross_val_score(model, X, Y, cv=10)
    print(scores)
    print('avg = %f' % np.mean(scores))

    print('[%s] Multi-Layer Perceptron:' % time.strftime('%Y-%m-%d %H:%M:%S'))
    model = MLPClassifier()
    scores = cross_val_score(model, X, Y, cv=10)
    print(scores)
    print('avg = %f' % np.mean(scores))


class MyDataset:
    def __init__(self, feature_type, file_list=None, labels=None, sub_features=None, dataframe_keys=None, assign_market=None):
        # 指定apk文件路径
        if file_list:
            self.file_list = file_list
            self.labels = labels

        # 不指定文件路径则记录根目录下所有apk文件路径，并过滤无效文件、记录label
        else:
            # self.file_list, self.labels = get_part_feature_files_and_labels(1000, 1000)  # TODO tmp
            self.file_list, self.labels = get_all_feature_files_and_labels()
        if self.labels:
            self.pos_count = self.labels.count(1)
            self.neg_count = len(self.labels) - self.pos_count
            self.files_train, self.files_test, self.labels_train, self.labels_test = train_test_split(self.file_list, self.labels, random_state=random_state)
        self.feature_type = feature_type
        self.feature_list = model2features[feature_type]

        if dataframe_keys:
            self.sub_features = sub_features
            self.dataframe_keys = dataframe_keys
        else:
            self.sub_features = {}
            self.dataframe_keys = []
            if self.feature_type == MODEL_CATE:
                if assign_market:
                    self.sub_features[self.feature_type] = category_mappings[assign_market]
                    self.dataframe_keys += category_mappings[assign_market].keys()
                else:
                    self.sub_features[self.feature_type] = category_mappings[market]
                    self.dataframe_keys += category_mappings[market].keys()
            elif self.feature_type == MODEL_ADS:
                self.dataframe_keys += self.feature_list
            elif self.feature_type == MODEL_RATE:
                self.sub_features['minAge'] = [-1, 3, 7, 12, 16, 18]
                self.dataframe_keys += [-1, 3, 7, 12, 16, 18]

    # 对规模大的feature，建立一个子feature映射，只考虑在映射中的子feature
    def build_sub_feature_map(self, ratio):
        # if self.feature_type != MODEL_CATE and self.feature_type != MODEL_ADS:
        if self.feature_type != MODEL_ADS and self.feature_type != MODEL_RATE:
            # 统计某个feature中每种子feature的出现频率，之后正式统计时只考虑这些出现频率多的
            for each_feature in self.feature_list:
                if each_feature in features_to_count:  # manifest model & code model will enter this branch
                    self.sub_features[each_feature] = self.get_popular_features_by_files(each_feature, ratio)
                    self.dataframe_keys += self.sub_features[each_feature]

    # 根据特征名，返回统计及排序后出现的该类特征所出现的特征值的list
    def get_popular_features_by_files(self, feature_name, ratio):
        if feature_name == 'category':
            return self.all_possible_sub(feature_name)
        # 正例负例分别统计子特征，取出现频率前30%的子特征，然后取交集
        # f_kid, f_adt = self.count_sub_features_all(feature_name)
        # f_kid = top_data_filter(f_kid, self.pos_count, ratio)
        # f_adt = top_data_filter(f_adt, self.neg_count, ratio)
        # final_features = [x for x in f_kid if x in f_adt]
        # 在所有应用上统计子特征，取出现频率前30%的子特征
        counter = self.count_sub_features_all(feature_name)
        if feature_name == 'apis':
            # 过滤选定的api
            # final_features = top_data_filter(counter, self.pos_count + self.neg_count, 0.5)
            # final_features = delete_apis_manually(final_features)
            # 过滤次数过多的api（儿童应用和成人应用中都常出现）
            # final_features = top_data_filter(counter, self.pos_count + self.neg_count, 0.2)
            # f_kid, f_adt = self.count_sub_features_all_two_class(feature_name)
            # f_kid = top_data_filter(f_kid, self.pos_count, 0.5)
            # f_adt = top_data_filter(f_adt, self.neg_count, 0.5)
            # common_features = [x for x in f_kid if x in f_adt]
            # print(len(final_features), len(f_kid), len(f_adt), len(common_features))
            # final_features2 = top_data_filter2(counter, self.pos_count + self.neg_count, 0.5)
            # final_features = [x for x in final_features if x not in common_features]
            # 选出与危险权限相关的api
            final_features = top_data_filter(counter, self.pos_count + self.neg_count, ratio)
            # final_features = select_mapped_apis(final_features, self)
            print('feature size =', len(final_features))
        else:
            final_features = top_data_filter(counter, self.pos_count + self.neg_count, ratio)

        # print('[%s] counted feature: %s\n%s' % (time.strftime('%Y-%m-%d %H:%M:%S'), feature_name, final_features))
        return final_features

    def all_possible_sub(self, feature_name):
        features = []
        for path in self.file_list:
            apk_features = read_apk_features(path)
            if apk_features[feature_name] not in features:
                features.append(apk_features[feature_name])
        return features

    # 根据给定特征名，统计所有apk中所有子特征出现次数
    def count_sub_features_all(self, feature_name):
        feature_count = {}
        # feature_count_adt = {}
        for path in self.file_list:
            apk_features = read_apk_features(path)
            count_sub_data(feature_count, apk_features, feature_name)
            # if apk_features['minAge'] < 12:
            #     count_sub_data(feature_count_kid, apk_features, feature_name)
            # else:
            #     count_sub_data(feature_count_adt, apk_features, feature_name)
        return feature_count  # _kid, feature_count_adt

    # 根据给定特征名，分别统计两类apk中所有子特征出现次数
    def count_sub_features_all_two_class(self, feature_name):
        feature_count_kid = {}
        feature_count_adt = {}
        for path, label in zip(self.file_list, self.labels):
            apk_features = read_apk_features(path)
            count_sub_data(feature_count_kid, apk_features, feature_name)
            if label == 1:
                count_sub_data(feature_count_kid, apk_features, feature_name)
            else:
                count_sub_data(feature_count_adt, apk_features, feature_name)
        return feature_count_kid, feature_count_adt

    # convert apk feature lists to dataframe under the relevant feature
    def convert_files_to_dataframe(self):
        all_apks_features = []

        for f in self.file_list:
            # print('[%s] reading file %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), f))
            apk = read_apk_features(f)

            apk_features = []
            for feature_name in self.feature_list:
                # print(feature_name)
                # 列表类型，只取部分
                if feature_name in features_to_count:
                    apk_features += sub_features_encoder(self.sub_features, feature_name, apk)

                # 应用类别
                elif type(apk[feature_name]) == str:
                    apk_features += category_encoder(self.sub_features, feature_name, apk)

                # 直接用的，布尔/数字（不需要用到feature_map）
                else:
                    # 由于feature_extractor中代码的疏忽，有loc为None的情况
                    if apk[feature_name] is None:
                        apk_features.append(0)
                    else:
                        apk_features.append(apk[feature_name])
                # print(len(apk_features))
                # print(apk_features)

            all_apks_features.append(apk_features)
        return pd.DataFrame(all_apks_features, columns=self.dataframe_keys)


def get_trained_model(model_name, file_list, label_list, assign_market):
    dataset = MyDataset(model_name, file_list, label_list, assign_market=assign_market)
    print('\nfeature to process: %s' % ', '.join(dataset.feature_list))

    print('[%s] counting features ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    if model_name != MODEL_CODE:
        dataset.build_sub_feature_map(0)
    else:
        dataset.build_sub_feature_map(0.6)

    print('[%s] there are %d apps suitable for kids, %d apps not suitable for kids'
          % (time.strftime('%Y-%m-%d %H:%M:%S'), dataset.pos_count, dataset.neg_count))

    print('\n[%s] reading files ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    X = dataset.convert_files_to_dataframe()
    Y = label_list

    # set different classifier & args for different model
    if model_name == MODEL_MANI:
        classifier = LogisticRegression()
        size = feature_size_assigned[model_name]
    elif model_name == MODEL_CODE:
        classifier = RandomForestClassifier()
        size = feature_size_assigned[model_name]
    else:
        size = 0
        if model_name == MODEL_CATE:
            classifier = BernoulliNB()
        else:
            classifier = DecisionTreeClassifier()

    print('\n[%s] selecting top%d features from %d features...' % (time.strftime('%Y-%m-%d %H:%M:%S'), size, X.shape[1]))
    X_filtered = select_best_chi2(X, Y, 100, size)
    print('[%s] done' % time.strftime('%Y-%m-%d %H:%M:%S'))

    print('\n[%s] training models ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    classifier.fit(X_filtered, label_list)
    print('[%s] done' % time.strftime('%Y-%m-%d %H:%M:%S'))

    if model_name == MODEL_ADS or model_name == MODEL_RATE:
        return classifier  # , dataset.sub_features, dataset.dataframe_keys
    elif model_name == MODEL_CATE:
        return classifier, dataset.sub_features, dataset.dataframe_keys
    columns_filtered = list(X_filtered.columns.values)
    mapping_filtered = {}
    for key, val in dataset.sub_features.items():
        mapping_filtered[key] = []
        for v in val:
            if v in columns_filtered:
                mapping_filtered[key].append(v)
    return classifier, mapping_filtered, columns_filtered


def predict_proba(model_name, file_list, classifier, args, assign_market):
    if args:
        dataset = MyDataset(model_name, file_list, sub_features=args[0], dataframe_keys=args[1])
    else:
        dataset = MyDataset(model_name, file_list, assign_market=assign_market)
    X = dataset.convert_files_to_dataframe()
    return classifier.predict_proba(X)


def explain_model(name, model, val, file, label):
    import dill
    # import joblib, json
    # model = joblib.load('%s/model_%s.pkl' % (market, name))
    # if os.path.exists('%s/vals_%s.json' % (market, name)):
    #     with open('%s/vals_%s.json' % (market, name), "rb") as handle:
    #         val = json.load(handle)
    file_exp = '%s/exp_%s.pkl' % (market, name)
    if not os.path.exists(file_exp):
        dataset = MyDataset(name)
        dataset = MyDataset(name, file_list=dataset.files_train, labels=dataset.labels_train, sub_features=val[0], dataframe_keys=val[1])
        X = dataset.convert_files_to_dataframe()
        explainer = LimeTabularExplainer(np.array(X), feature_names=X.columns, class_names=['nkid', 'kid'])
        with open(file_exp, 'wb') as f:
            dill.dump(explainer, f)
    else:
        with open(file_exp, 'rb') as f:
            explainer = dill.load(f)
    dataset = MyDataset(name, file_list=[file], sub_features=val[0], dataframe_keys=val[1])
    x = dataset.convert_files_to_dataframe()
    for _, x in x.iterrows():
        exp = explainer.explain_instance(x, model.predict_proba, num_features=50)
        print('true label =', ['nkid', 'kid'][label])
        print('prob(kid) =', model.predict_proba(np.array(x).reshape(1, -1))[0, 1])
        for e in exp.as_list():
            print(e)


# def explain_model(model_name, classifier, file, label, args):
#     if args:
#         dataset = MyDataset(model_name, [file], sub_features=args[0], dataframe_keys=args[1])
#     else:
#         dataset = MyDataset(model_name, [file])
#     X = dataset.convert_files_to_dataframe()
#
#     explainer = LimeTabularExplainer(X, class_names=['adt', 'kids'])
#     exp = explainer.explain_instance(vector, classifier.predict_proba, num_features=6)
#     print('true label = ', ['adt', 'kids'][label])
#     print('prob(kid) = ', classifier.predict_proba([data])[0, 1])
#     print(exp.as_list())
#     # exp.save_to_file('exp.html', text=True)


def train_callee(feature_name, file_list, labels, grid):
    dataset = MyDataset(feature_name, file_list, labels)
    print('feature to process: %s' % ', '.join(dataset.feature_list))
    print('\n[%s] counting features ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    dataset.build_sub_feature_map(0)  # 0.3指取前70%
    print('[%s] there are %d apps suitable for kids, %d apps not suitable for kids'
          % (time.strftime('%Y-%m-%d %H:%M:%S'), dataset.pos_count, dataset.neg_count))
    print('\n[%s] reading files ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    X = dataset.convert_files_to_dataframe()
    Y = dataset.labels
    ratio = 100
    print('\n[%s] selecting top%d%% features from %d features...' % (
    time.strftime('%Y-%m-%d %H:%M:%S'), ratio, X.shape[1]))
    if feature_name == MODEL_MANI:
        X_filtered = select_best_chi2(X, Y, ratio, 100)
    else:
        X_filtered = select_best_chi2(X, Y, ratio)
    if grid:
        print('\n[%s] grid training models ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
        train_models_grid(X_filtered, Y)
    else:
        print('\n[%s] default training models ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
        train_default_models(X_filtered, Y)
    print('[%s] done\n\n' % time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    # explain_model()
    # dataset = MyDataset(MODEL_MANI)
    # train_callee(MODEL_MANI, dataset.files_train, dataset.labels_train)
    feature_type = [MODEL_MANI, MODEL_CODE, MODEL_CATE, MODEL_ADS, MODEL_RATE][4]
    dataset = MyDataset(feature_type)
    dataset = MyDataset(feature_type, file_list=dataset.files_train, labels=dataset.labels_train)

    print('feature to process: %s' % ', '.join(dataset.feature_list))

    print('\n[%s] counting features ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    if feature_type != MODEL_CODE:
        dataset.build_sub_feature_map(0)  # 0.3指取top70%（是最高值的前70%，而不是数量），数字越大保留的特征越少
    else:
        dataset.build_sub_feature_map(0.6)  # 0.3指取top70%（是最高值的前70%，而不是数量），数字越大保留的特征越少
    # exit()

    print('[%s] there are %d apps suitable for kids, %d apps not suitable for kids'
          % (time.strftime('%Y-%m-%d %H:%M:%S'), dataset.pos_count, dataset.neg_count))

    print('\n[%s] reading files ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    X = dataset.convert_files_to_dataframe()
    Y = dataset.labels

    # if features_processing == features_in_number:
    #     print('raw data:\n', X)
    #     # 数据标准化
    #     scaler = MinMaxScaler()
    #     X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns.values)
    #     # X = minmax_scale(X, axis=0)
    #     print('standardized data:\n', X)

    ratio_list = [100]
    # size_list = [25, 50, 100, 160]
    # if feature_type == MODEL_CODE:
    #     ratio_list = [2, 5, 10]
    if feature_type in feature_size_assigned:
        size = feature_size_assigned[feature_type]
    else:
        size = 0
    for ratio in ratio_list:
        print('\n[%s] selecting top%d features from %d features...' % (time.strftime('%Y-%m-%d %H:%M:%S'), size, X.shape[1]))
        X_filtered = select_best_chi2(X, Y, ratio, size)
        # libsvm_transformer(X_filtered.values, Y, 'data_libsvm_%s_%d_%s_%d.txt' % (market, 1000*2, feature_type, ratio))
        print('[%s] done' % time.strftime('%Y-%m-%d %H:%M:%S'))

        # print('\n[%s] training models ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
        # # train_models_grid(X_filtered, Y)
        train_default_models(X_filtered, Y)
        # print('[%s] done' % time.strftime('%Y-%m-%d %H:%M:%S'))
