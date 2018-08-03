import pickle
import codecs
import jieba
import math


def load_word2vec(path):
    from gensim import models
    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    wv = model.wv
    del model
    return wv


def load_stopwords(path):
    stopwords = {}.fromkeys([line.strip() for line in codecs.open(path, 'r', encoding='utf8')])
    return stopwords


def build_idf(input_file, output_file):
    stopwords = load_stopwords('../resources/stopwords.txt')
    idf = {}
    count = 0
    with open(input_file, 'r') as f:
        pre_q = ''
        for line in f:
            items = line.split('\t')
            q, a = items[0], items[1]
            a_no_stop = [word for word in jieba.cut(a) if word not in stopwords]
            existed = False
            for word in a_no_stop:
                if word not in idf:
                    idf[word] = 1
                elif not existed:
                    idf[word] += 1
                    existed = True
            count += 1
            if q != pre_q and len(pre_q) > 0:
                count += 1
                q_no_stop = [word for word in jieba.cut(pre_q) if word not in stopwords]
                existed = False
                for word in q_no_stop:
                    if word not in idf:
                        idf[word] = 1
                    elif not existed:
                        idf[word] += 1
                        existed = True
            pre_q = q
        # handle the last question
        count += 1
        q_no_stop = [word for word in jieba.cut(pre_q) if word not in stopwords]
        existed = False
        for word in q_no_stop:
            if word not in idf:
                idf[word] = 1
            elif not existed:
                idf[word] += 1
                existed = True
    for key, value in idf.items():
        idf[key] = math.log(count / value, 10)
    pickle.dump(idf, open(output_file, 'wb'), True)


def load_idf(file_path):
    return pickle.load(open(file_path, 'rb'))


if __name__ == '__main__':
    # build_idf('../data/small/nlpcc-iccpol-2016.dbqa.training-data', '../resources/idf.train.pkl')
    build_idf('../data/test.txt', '../resources/idf.test.pkl')

