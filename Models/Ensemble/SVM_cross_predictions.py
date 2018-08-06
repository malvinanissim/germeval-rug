'''
This script is to be used for the ensemble system and retrieves predictions for a given set of samples by the SVM.

This SVM needs 1 dataset, using cross validation to output a value for each X
predictions stored in pickle
'''
import argparse
import re
import statistics as stats
import stop_words
import json
import pickle
import gensim.models as gm
import numpy as np
from scipy.sparse import hstack, csr_matrix

import features
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion


def read_corpus(corpus_file, binary=True):
    '''Reading in data from corpus file'''

    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # making sure no missing labels
            if len(data) != 3:
                raise IndexError('Missing data for tweet "%s"' % data[0])

            tweets.append(data[0])

            if binary:
                # 2-class problem: OTHER vs. OFFENSE
                labels.append(data[1])
            else:
                # 4-class problem: OTHER, PROFANITY, INSULT, ABUSE
                labels.append(data[2])

    return tweets, labels

def read_corpus_binary(pos_file, neg_file, pos_label, neg_label):
    '''Reading in data from 2 files, containing the positive and the negative training samples
    Order: All positive samples first, then all negative samples'''

    X, Y = [],[]
    # Getting all positive samples
    with open(pos_file, 'r', encoding='utf-8') as fpos:
        for line in fpos:
            assert len(line) > 0, 'Empty line found!'
            X.append(line.strip())
            Y.append(pos_label)
    # Getting all negative samples
    with open(neg_file, 'r', encoding='utf-8') as fneg:
        for line in fneg:
            assert len(line) > 0, 'Empty line found!'
            X.append(line.strip())
            Y.append(neg_label)

    print('len(X):', len(X))
    print('len(Y):', len(Y))

    return X, Y



def load_embeddings(embedding_file):
    '''
    loading embeddings from file
    input: embeddings stored as json (json), pickle (pickle or p) or gensim model (bin)
    output: embeddings in a dict-like structure available for look-up, vocab covered by the embeddings as a set
    '''
    if embedding_file.endswith('json'):
        f = open(embedding_file, 'r', encoding='utf-8')
        embeds = json.load(f)
        f.close
        vocab = {k for k,v in embeds.items()}
    elif embedding_file.endswith('bin'):
        embeds = gm.KeyedVectors.load(embedding_file).wv
        vocab = {word for word in embeds.index2word}
    elif embedding_file.endswith('p') or embedding_file.endswith('pickle'):
        f = open(embedding_file,'rb')
        embeds = pickle.load(f)
        f.close
        vocab = {k for k,v in embeds.items()}

    return embeds, vocab


if __name__ == '__main__':


    X, Y = read_corpus('../../Data/germeval.ensemble.train.txt')
    assert len(X) == len(Y), 'Unequal length for X and Y!'
    print('len(train_dat):', len(X))

    # X = X[:30]
    # Y = Y[:30]

    # Vectorizing data / Extracting feature
    # unweighted word uni and bigrams
    count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('de'))
    count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))

    # Getting twitter embeddings
    # path_to_embs = '../../Resources/test_embeddings.json'
    path_to_embs = '../embeddings/twitter_de_52D.p'
    print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
    embeddings, _ = load_embeddings(path_to_embs)
    print('Done')

    vectorizer = FeatureUnion([('word', count_word),
                                ('char', count_char),
                                ('word_embeds', features.Embeddings(embeddings, pool='max'))])

    classifier = Pipeline([
                            ('vectorize', vectorizer),
                            ('classify', SVC(kernel='linear', probability=True))
    ])

    folds = 5
    print('Cross-validating on {} folds...'.format(folds))
    Yguess = cross_val_predict(classifier, X, Y, cv=folds, method='predict_proba')


    print('Turning to scipy:')
    Ysvm = csr_matrix(Yguess)
    print(type(Ysvm))
    print(Ysvm.shape)
    print(Ysvm)


    # Pickling the predictions
    save_to = open('NEW-train-svm-predict.p', 'wb')
    pickle.dump(Ysvm, save_to)
    save_to.close()
