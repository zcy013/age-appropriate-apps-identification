#!/usr/bin/python3
# encoding: utf-8

import os
import json
import re
import time
import jieba
import numpy as np
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from snownlp import SnowNLP
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE, SelectKBest, SelectPercentile, chi2

from configs import random_state, feature_size_assigned, MODEL_TEXT, MODEL_NAME, MODEL_OCRS
from utils import read_apk_features, get_all_feature_files_and_labels, train_default_models

stemmer = SnowballStemmer("english")  # 词干提取
lemmatizer = WordNetLemmatizer()  # 词型还原


def load_stop_words(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return set(line.strip() for line in lines)


stopwords_en = set(stopwords.words('english'))
stopwords_cn = load_stop_words('./stop_words.txt')


def read_apk_features(path):
    apk_feature = []
    try:
        with open(path, "rb") as handle:
            apk_feature = json.load(handle)
    except Exception as e:
        print('error on reading %s: %s' % (path, e))
    return apk_feature


def extract_string_feature(string_feature):
    pattern = re.compile(r'<string.*>(.*)</string>')
    matched = re.findall(pattern, string_feature)
    return ' '.join(matched)


def text_pre_processing(text):
    # 1. 分词
    words_raw = jieba.lcut(text)

    # 2. 清洗（符号，停用词等）
    # 3. 标准化（对英文：大写转小写，词形还原、词干提取；对中文：繁体转简体）
    words_clean = []
    for i in range(len(words_raw)):
        word = words_raw[i]

        # 中文
        if re.findall('[\u4e00-\u9fa5]+', word):
            word = SnowNLP(word).han
            # 停用词
            if word in stopwords_cn:
                continue
            words_clean.append(word)

        # TODO 按中文分词破坏了英文的结构怎么办？
        # 英文
        elif re.findall('[a-zA-Z]+', word):
            word = word.lower()  # TODO 如果是大写的专有名词怎么办？
            # 停用词
            if word in stopwords_en or len(word) == 1:  # 过滤单个字母？TODO
                continue
            word = stemmer.stem(word)
            # word = lemmatizer.lemmatize(word)
            if len(word) > 1:  # 过滤单个字母？TODO
                words_clean.append(word)

        elif re.findall('[0-9]+', word):
            # 过滤短数字？TODO
            if len(word) > 5:
                words_clean.append(word)

        # 其他情况为符号，其他语言等，均过滤掉
    return words_clean


class MyCorpus:
    def __init__(self, feature_type, file_list=None, labels=None):
        self.feature_type = feature_type
        self.key_words = []
        # 指定apk文件路径
        if file_list:
            self.file_list = file_list
            self.labels = labels
        # 不指定文件路径则记录根目录下所有apk文件路径，并过滤无效文件、记录label
        else:
            self.file_list, self.labels = get_all_feature_files_and_labels()
        if self.labels:
            self.files_train, self.files_test, self.labels_train, self.labels_test = train_test_split(self.file_list, self.labels, random_state=random_state)

        self.avg_len = np.zeros((len(self.file_list), 1))

        self.library = []
        counter = 0
        for file in self.file_list:
            if feature_type == 'ocr':
                imgs = read_apk_features(file.replace('features_all', 'features_img'))

                ocrs = []
                for img, features in imgs.items():
                    if re.match('^screenshot\d[\.][a-z]+$', img):  # 只考虑截图中的string
                        if 'string_en' in features:
                            ocrs.append(features['string_en'])
                        else:
                            ocrs.append(features['string'])
                try:
                    self.avg_len[counter][0] = len(''.join(ocrs)) / len(ocrs)
                    self.library.append(' '.join(text_pre_processing(' '.join(ocrs))))
                except:
                    self.avg_len[counter][0] = 0
                    self.library.append('')
            else:
                apk = read_apk_features(file)
                if feature_type + '_en' in apk:
                    self.library.append(' '.join(text_pre_processing(apk[feature_type + '_en'])))
                else:
                    self.library.append(' '.join(text_pre_processing(apk[feature_type])))
            counter += 1


def select_best_chi2(X, Y, feature_names, num):
    model = SelectKBest(chi2, k=num)
    # model = SelectPercentile(chi2, percentile=ratio)
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
        print('%d\t%s\t%f\t%f' % (i + 1, feature_names[index], model.scores_[index], model.pvalues_[index]))
        # print('%d\t%s\t%f' % (i + 1, X.columns.values[index], model.scores_[index]))
    # X_new = pd.DataFrame(X_new, columns=X.columns.values[model.get_support(indices=True)])
    columns = [feature_names[i] for i in model.get_support(indices=True)]
    return X_new, columns


def build_vec():
    vectorizer = TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    rf.fit(train_vectors, y_train)

    pred = rf.predict(test_vectors)
    sklearn.metrics.f1_score(y_test, pred, average='binary')

    return vectorizer, rf


def fit_transform(feature_type, file_list, labels):
    print('\n[%s] pre-processing ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    corpus = MyCorpus(feature_type, file_list, labels)
    vectorizer = CountVectorizer(lowercase=False)
    X = vectorizer.fit_transform(corpus.library)
    Y = corpus.labels
    X, corpus.key_words = select_best_chi2(X, Y, vectorizer.get_feature_names(),
                                           10 if feature_type == 'description' else 5)
    print('[%s] keywords: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), corpus.key_words))
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    if feature_type != 'ocr':
        return corpus.key_words, transformer, tfidf.toarray()
    else:
        return corpus.key_words, transformer, np.concatenate((tfidf.toarray(), corpus.avg_len), axis=1)
    # return pd.DataFrame(tfidf.toarray(), columns=corpus.key_words)


def transform(feature_type, file_list, keywords, transformer):
    print('\n[%s] pre-processing ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    corpus = MyCorpus(feature_type, file_list)
    vectorizer = CountVectorizer(lowercase=False, vocabulary=keywords)
    X = vectorizer.transform(corpus.library)
    tfidf = transformer.transform(X)
    if feature_type != 'ocr':
        return tfidf.toarray()
    else:
        return np.concatenate((tfidf.toarray(), corpus.avg_len), axis=1)


def explain_model(vectorizer, classifier, data, label):
    pipeline = make_pipeline(vectorizer, classifier)
    explainer = LimeTextExplainer(class_names=['adt', 'kid'])
    exp = explainer.explain_instance(data, pipeline.predict_proba, num_features=6)

    print('\nprob(kid) = ', pipeline.predict_proba([data])[0, 1])
    print('true label = ', ['adt', 'kid'][label])
    print(exp.as_list())

    exp.save_to_file('exp.html', text=True)


def get_keywords(X, Y, n):
    tfidf_vectorizer = TfidfVectorizer(max_features=n)
    tfidf = tfidf_vectorizer.fit_transform(X)
    X_filtered, key_words = select_best_chi2(tfidf, Y, tfidf_vectorizer.get_feature_names(), n)
    print(tfidf_vectorizer.get_feature_names())
    return X_filtered, key_words, tfidf_vectorizer


def get_trained_model(model_name, file_list, labels_true):
    corpus = MyCorpus(model_name, file_list=file_list, labels=labels_true)
    X_filtered, keywords, vectorizer = get_keywords(corpus.library, corpus.labels, feature_size_assigned[model_name])
    classifier = BernoulliNB()
    classifier.fit(X_filtered, labels_true)
    return make_pipeline(vectorizer, classifier)


def predict_proba(model_name, file_list, pipeline):
    corpus = MyCorpus(model_name, file_list=file_list)
    return pipeline.predict_proba(corpus.library)


if __name__ == '__main__':
    # name, description, ocr
    feature_type_code = 0
    feature_type = ['name', 'description', 'ocr'][feature_type_code]
    feature_TYPE = [MODEL_NAME, MODEL_TEXT, MODEL_OCRS][feature_type_code]
    size = feature_size_assigned[feature_TYPE]

    # print('\n[%s] pre-processing ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    corpus = MyCorpus(feature_type)
    corpus = MyCorpus(feature_type, file_list=corpus.files_train, labels=corpus.labels_train)
    X_filtered, keywords, _ = get_keywords(corpus.library, corpus.labels, size)
    train_default_models(X_filtered, corpus.labels)

    # vectorizer = CountVectorizer()
    # X = vectorizer.fit_transform(corpus.library)
    # Y = corpus.labels
    # X_filtered, corpus.key_words = select_best_chi2(X, Y, vectorizer.get_feature_names(),
    #                                                 10 if feature_type == 'description' else 5)
    # print('[%s] got keywords for feature %s: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), feature_type, corpus.key_words))
    # transformer = TfidfTransformer()
    # tfidf = transformer.fit_transform(X_filtered)
    #
    # X_train, X_test, y_train, y_test = train_test_split(corpus.library, corpus.labels)
    # print('[%s] done' % time.strftime('%Y-%m-%d %H:%M:%S'))
    #
    # print('\n[%s] training model ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    # vectorizer, classifier = build_vec()
    # print('[%s] done' % time.strftime('%Y-%m-%d %H:%M:%S'))
    #
    # for i in range(len(X_test)):
    #     explain_model(vectorizer, classifier, X_test[i], y_test[i])
    # for i in range(len(X_train)):
    #     explain_model(vectorizer, classifier, X_train[i], y_train[i])
