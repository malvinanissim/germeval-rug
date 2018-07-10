'''
This script is to be used for the ensemble system and retrieves predictions for a given set of samples by the SVM
It learns and performs the multi-class classification task, but at the end we map it's findings of OTHER to 0 and
everything else to 1.
It attempts to improve the ensemble classifier, which is only binary.

This SVM needs 2 datasets, train and test, training itself on the trainset and outputting predictions for each X in testset
Predictions stored in pickle
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

def map_to_binary(predictions):
    '''
    The function maps the output of the multi-class classifier to binary classes.
    OTHER to 0, everything else to 1
    Takes list of predictions, returns a mapped np.array of predictions of shape (len(predictions), 1)
    '''
    mapped = []
    for guess in predictions:
        if guess == 'OTHER':
            mapped.append(0)
        else:
            mapped.append(1)
            # print('Found', guess)

    return np.array(mapped).reshape((len(mapped),1))



if __name__ == '__main__':

    # load training data set
    # pos_path = '../../Data/offense.train.txt'
    # neg_path = '../../Data/other.train.txt'

    Xtrain, Ytrain = read_corpus('../Data/germeval.ensemble.train.txt', binary=False)
    assert len(Xtrain) == len(Ytrain), 'Unequal length for Xtrain and Ytrain!'
    print('{} train samples'.format(len(Xtrain)))

    # load testing data set
    # pos_path = '../../Data/offense.test.txt'
    # neg_path = '../../Data/other.test.txt'

    Xtest, Ytest = read_corpus('../Data/germeval.ensemble.test.txt', binary=False)
    assert len(Xtest) == len(Ytest), 'Unequal length for Xtest and Ytest!'
    print('{} test samples'.format(len(Xtest)))

    # Xtrain, Ytrain = Xtrain[:60], Ytrain[:60]

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
                            ('classify', SVC(kernel='linear'))
    ])

    print('Fitting model...')
    classifier.fit(Xtrain, Ytrain)

    print('Predicting...')
    Yguess = classifier.predict(Xtest)

    # Map to binary labels
    Yguess = map_to_binary(Yguess)
    # print(Yguess.shape)
    # print(Yguess[:20])

    print('Turning to scipy:')
    Ysvm = csr_matrix(Yguess)
    print(type(Ysvm))
    print(Ysvm.shape)
    print(Ysvm)

    # Pickling the predictions
    save_to = open('Multi-test-predict.p', 'wb')
    pickle.dump(Ysvm, save_to)
    save_to.close()
