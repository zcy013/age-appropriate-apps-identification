#!/usr/bin/python3
# encoding: utf-8

import os
import re
import time
import jieba
import numpy as np

from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from snownlp import SnowNLP
from gensim.models.word2vec import Word2Vec
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import MinMaxScaler
from bert_serving.client import BertClient

from utils import read_apk_features, get_all_feature_files_and_labels, get_part_feature_files_and_labels, train_models_grid, train_default_models
from configs import features_text, random_state, market

stemmer = SnowballStemmer("english")  # 词干提取
lemmatizer = WordNetLemmatizer()  # 词型还原
if market == 'google':
    bert_client = BertClient(ip='10.20.38.233', port=5577, port_out=5599)  # eng
else:
    bert_client = BertClient(ip='10.20.38.233', port=5566, port_out=5500)  # chi


def load_stop_words(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return set(line.strip() for line in lines)


stopwords_en = set(stopwords.words('english'))
stopwords_cn = load_stop_words('./stop_words.txt')


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
    # 为节省内存，不把语料库存入变量，而是每次用到时从文件读取
    def __init__(self, vector_size, file_list=None, labels=None, kid_count=None, adt_count=None):
        self.vectorizer = None
        self.vectors = None
        self.vector_size = vector_size
        # 指定apk文件路径
        if file_list:
            self.file_list = file_list
            self.labels = labels
        # 不指定文件路径则记录根目录下所有apk文件路径，并过滤无效文件、记录label
        elif kid_count and adt_count:
            self.file_list, self.labels = get_part_feature_files_and_labels(kid_count, adt_count)
        else:
            self.file_list, self.labels = get_all_feature_files_and_labels()
        if self.labels:
            self.pos_count = self.labels.count(1)
            self.neg_count = len(self.labels) - self.pos_count
            self.files_train, self.files_test, self.labels_train, self.labels_test = train_test_split(self.file_list, self.labels, random_state=random_state)
        self.library = []

    def __iter__(self):
        for path in self.file_list:
            apk = read_apk_features(path)
            # print('[%s] reading file %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), path))
            text = []
            # 文本类，合在一起
            for feature_name in features_text:
                if feature_name != 'strings':
                    text.append(apk[feature_name])
                else:
                    text.append(extract_string_feature(apk['strings']))

            # 节省内存，但因为训练模型和计算句向量分别需要遍历一次library，耗时会变为两倍 TODO
            # 每次在外部遍历library，内部会遍历iter+1遍
            yield text_pre_processing(' '.join(text))

    def __getitem__(self, item):
        apk = read_apk_features(self.file_list[item])
        print('\n[%s] reading file %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.file_list[item]))
        text = []
        # 文本类，合在一起
        for feature_name in features_text:
            if feature_name != 'strings':
                text.append(apk[feature_name])
            else:
                text.append(extract_string_feature(apk['strings']))
        return ' '.join(text_pre_processing(' '.join(text)))

    # 用于bert的输入（不需要分词），其他函数适用于word2vec
    def get_original_sentences(self):
        if not self.library:
            for path in self.file_list:
                apk = read_apk_features(path)
                # print('[%s] reading file %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), path))
                text = []
                # 文本类，合在一起
                for feature_name in features_text:
                    if feature_name != 'strings':
                        if feature_name + '_en' in apk:
                            text.append(apk[feature_name + '_en'])
                        else:
                            text.append(apk[feature_name])
                    else:
                        text.append(extract_string_feature(apk['strings']))
                if not ' '.join(text).strip():
                    print(path, text)
                self.library.append(' '.join(text))
        return self.library

    def build_vectorizer_w2c(self):
        if not self.vectorizer:
            # TODO 数据集切分？train输入模型test用来train
            model = Word2Vec(iter=1, size=self.vector_size, workers=20, min_count=5, sg=1)
            model.build_vocab(self)  # 第一次遍历library
            model.train(self, total_examples=model.corpus_count, epochs=model.epochs)  # 第二次遍历library
            # model.save('./model_text.pkl')  # 保存模型
            # model = Word2Vec.load('./text_model.pkl')  # 调用模型
            self.vectorizer = Word2VecVectorizer(dict(zip(model.wv.index2word, model.wv.syn0)), self.vector_size)
            # vectorizer字典中每个键是模型选择出的keywords？数量是学习出来的，=model.vocab
            # 值是键的向量表达，值的长度=model.size
        return self.vectorizer

    def build_sentence_vectors(self):
        if not self.vectors:
            # 两种方法等效
            # vectors = build_sentence_vectors(self, vector_size, model)
            if not self.vectorizer:
                self.build_vectorizer_w2c()
            self.vectors = self.vectorizer.transform(self)  # 第三次遍历library
            np.save('train_vecs.npy', self.vectors)
            # vectors = np.load('train_vecs.npy')
        return self.vectors


# 对每个句子的所有词向量取均值，来生成一个句子的vector
# def build_sentence_vectors(library, size, model):
#     avg_data = []
#     for text in library:
#         vec = np.zeros(size)
#         count = 0
#         for word in text:
#             try:
#                 # tmp = model[word]
#                 # print(model[word])
#                 vec += model[word]
#                 count += 1
#             except KeyError:
#                 continue
#         if count != 0:
#             avg_data.append(vec / count)
#         else:
#             avg_data.append(vec)
#     return avg_data


# select features from Trees
def select_features_from_tree(X, Y, feature_size):
    print('[%s] training tree ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    model = ExtraTreesClassifier()
    scores = cross_val_score(model, X, Y, cv=10)
    print(scores)
    model.fit(X, Y)
    return model

    # model.fit(X, Y)
    # F_dict = dict(sorted(zip(X.columns, model.feature_importances_), reverse=True)[:feature_size])
    # print('[%s] got features: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), F_dict))
    # Y_ = model.predict(X)
    # print('[%s] accuracy = %f' % (time.strftime('%Y-%m-%d %H:%M:%S'), accuracy_score(Y, Y_)))
    # print('[%s] confusion matrix: \n%s' % (time.strftime('%Y-%m-%d %H:%M:%S'), confusion_matrix(Y, Y_)))
    # print('[%s] classification report: \n%s' % (time.strftime('%Y-%m-%d %H:%M:%S'), classification_report(Y, Y_)))
    # return list(F_dict.keys())


def select_best_chi2(X, Y, feature_size):
    model = SelectKBest(chi2, k=feature_size)
    X_new = model.fit_transform(X, Y)
    # print(model.scores_)  # 得分越高，特征越重要
    # print(model.pvalues_)  # p-values 越小，置信度越高，特征越重要
    # mask = model.get_support()  # 得到每个特征是否被选择的T-F数组
    # indices = model.get_support(indices=True)  # 得到所有选择出的特征的索引数组（按索引顺序）
    indices = np.argsort(model.scores_)[::-1]  # 得到所有选择出的特征的索引数组（按score倒序）
    # best_features = list(X.columns.values[indices[0:feature_size]])
    # for i in range(len(indices)):
    #     index = indices[i]
    #     if model.scores_[index] == 0:
    #         break
    #     print(i, X.columns.values[index], model.scores_[index])
    # X_new = pd.DataFrame(X_new, columns=X.columns.values[model.get_support(indices=True)])
    return X_new


# def explain_model(pipeline, data, label):
#     explainer = LimeTextExplainer(class_names=['adt', 'kids'])
#     exp = explainer.explain_instance(data, pipeline.predict_proba, num_features=6)
#     print('true label = ', ['adt', 'kids'][label])
#     print('prob(kid) = ', pipeline.predict_proba([data])[0, 1])
#     print(exp.as_list())
#     # exp.save_to_file('exp.html', text=True)


def explain_model(classifier, file, label):
    apk = read_apk_features(file)
    data = []
    # 文本类，合在一起
    for feature_name in features_text:
        if feature_name != 'strings':
            data.append(apk[feature_name])
        else:
            data.append(extract_string_feature(apk['strings']))
    data = ' '.join(data)
    # vector = bert_client.encode(data)
    pipeline = make_pipeline(BertVectorizer(bert_client), classifier)
    try:
        explainer = LimeTextExplainer(class_names=['nkid', 'kid'])
        exp = explainer.explain_instance(data, pipeline.predict_proba, num_features=10)
        print('true label =', ['nkid', 'kid'][label])
        print('prob(kid) =', pipeline.predict_proba([data])[0, 1])
        for e in exp.as_list():
            print(e)
    except:
        pass
    # exp.save_to_file('exp.html', text=True)


class Word2VecVectorizer(object):
    def __init__(self, word2vec, vector_size):
        self.word2vec = word2vec
        self.dim = vector_size

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]  # 句子中所有词向量的均值
                    or [np.zeros(self.dim)], axis=0)  # 有未知词则全设为0
            for words in X
        ])


class BertVectorizer(object):
    def __init__(self, client):
        self.client = client

    def fit(self, X, y):
        return self

    def transform(self, X):
        return self.client.encode(X)


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


def get_trained_model(file_list, labels_true):
    corpus = MyCorpus(100, file_list, labels_true)
    # print('\n[%s] building vectorizer ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    # vectorizer = corpus.build_vectorizer_w2c()

    print('\n[%s] getting vectors ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    # vectors = corpus.build_sentence_vectors()
    vectors = bert_client.encode(corpus.get_original_sentences())

    print('\n[%s] building classifier ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    classifier = SVC(probability=True)
    classifier.fit(vectors, labels_true)
    # print('feature importance:\n', classifier.feature_importances_)
    # labels_pred = classifier.predict(vectors)
    # print('accuracy = ', accuracy_score(labels_true, labels_pred))

    # return make_pipeline(vectorizer, classifier)
    return classifier


def predict_proba(file_list, classifier):
    corpus = MyCorpus(100, file_list)
    X = bert_client.encode(corpus.get_original_sentences())
    return classifier.predict_proba(X)


def train_callee(file_list, labels, grid):
    print('[%s] building dataset' % time.strftime('%Y-%m-%d %H:%M:%S'))
    corpus = MyCorpus(40, file_list, labels)
    library = corpus.get_original_sentences()
    print('\n[%s] getting vectors ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    vectors = bert_client.encode(library)
    Y = corpus.labels
    vectors = MinMaxScaler().fit_transform(vectors)
    print('\n[%s] building classifier ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    if grid:
        train_models_grid(vectors, Y)
    else:
        train_default_models(vectors, Y)


if __name__ == '__main__':
    print('[%s] building dataset' % time.strftime('%Y-%m-%d %H:%M:%S'))
    corpus = MyCorpus(0)
    corpus = MyCorpus(0, file_list=corpus.files_train, labels=corpus.labels_train)
    library = corpus.get_original_sentences()
    print('[%s] done, selected %d + %d samples:' % (time.strftime('%Y-%m-%d %H:%M:%S'), corpus.pos_count, corpus.neg_count))
    # for label in [0, 1]:
    #     for i in range(len(corpus.labels)):
    #         if corpus.labels[i] == label:
    #             print(label, library[i])
    #     print()

    # print('\n[%s] building vectorizer ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    # vectorizer = corpus.build_vectorizer_w2c()

    print('\n[%s] getting vectors ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    # vectors = corpus.build_sentence_vectors()
    vectors = bert_client.encode(library)
    Y = corpus.labels

    vectors = MinMaxScaler().fit_transform(vectors)
    # print('\n[%s] selecting features from %d features...' % (time.strftime('%Y-%m-%d %H:%M:%S'), vectors.shape[1]))
    # X_filtered = select_best_chi2(vectors, Y, min(200, vectors.shape[1]))
    # print('[%s] done' % time.strftime('%Y-%m-%d %H:%M:%S'))
    #
    # print('\n[%s] training model ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    # tree = select_features_from_tree(X_filtered, Y, 200)
    # print('[%s] done' % time.strftime('%Y-%m-%d %H:%M:%S'))

    print('\n[%s] building classifier ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    train_default_models(vectors, Y)
    # train_models_grid(vectors, Y)
    # classifier = RandomForestClassifier(n_estimators=500)
    # classifier.fit(vectors, Y)
    # print('[%s] done' % time.strftime('%Y-%m-%d %H:%M:%S'))
    # print('feature importance:\n', classifier.feature_importances_)
    # Y_ = classifier.predict(vectors)
    # print('accuracy = ', accuracy_score(Y, Y_))

    # pipeline = make_pipeline(vectorizer, classifier)
    # for i in range(len(Y)):
    #     explain_model(pipeline, corpus[i], Y[i])
    print('\n[%s] done' % time.strftime('%Y-%m-%d %H:%M:%S'))
