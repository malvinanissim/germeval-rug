'''
This script implements an ensemble classifer for GermEval 2018.
The lower-level classifiers are SVM and CNN
the meta-level classifer is optionally a LinearSVC or a Logistic Regressor

This is a second version, incorporating information from the multi-class SVM classifier as well!

Predictions outputted by SVM and CNN need to be obtained beforehand, stored as pickle, and loaded
'''

import argparse
import re
import statistics as stats
import stop_words
import features
import json
import pickle
from scipy.sparse import hstack, csr_matrix

# from features import get_embeddings
from sklearn.base import TransformerMixin
from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

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

# def read_corpus_binary(pos_file, neg_file, pos_label, neg_label):
#     '''Reading in data from 2 files, containing the positive and the negative training samples
#     Order: All positive samples first, then all negative samples'''
#
#     X, Y = [],[]
#     # Getting all positive samples
#     with open(pos_file, 'r', encoding='utf-8') as fpos:
#         for line in fpos:
#             assert len(line) > 0, 'Empty line found!'
#             X.append(line.strip())
#             Y.append(pos_label)
#     # Getting all negative samples
#     with open(neg_file, 'r', encoding='utf-8') as fneg:
#         for line in fneg:
#             assert len(line) > 0, 'Empty line found!'
#             X.append(line.strip())
#             Y.append(neg_label)
#
#     print('len(X):', len(X))
#     print('len(Y):', len(Y))
#
#     return X, Y


def evaluate(Ygold, Yguess):
    '''Evaluating model performance and printing out scores in readable way'''

    print('-'*50)
    print("Accuracy:", accuracy_score(Ygold, Yguess))
    print('-'*50)
    print("Precision, recall and F-score per class:")

    # get all labels in sorted way
    # Ygold is a regular list while Yguess is a numpy array
    labs = sorted(set(Ygold + Yguess.tolist()))

    # printing out precision, recall, f-score for each class in easily readable way
    PRFS = precision_recall_fscore_support(Ygold, Yguess, labels=labs)
    print('{:10s} {:>10s} {:>10s} {:>10s}'.format("", "Precision", "Recall", "F-score"))
    for idx, label in enumerate(labs):
        print("{0:10s} {1:10f} {2:10f} {3:10f}".format(label, PRFS[0][idx],PRFS[1][idx],PRFS[2][idx]))

    print('-'*50)
    print("Average (macro) F-score:", stats.mean(PRFS[2]))
    print('-'*50)
    print('Confusion matrix:')
    print('Labels:', labs)
    print(confusion_matrix(Ygold, Yguess, labels=labs))
    print()



if __name__ == '__main__':

    '''
    PART: TRAINING META-CLASSIFIER
    '''

    # load training data of ensemble classifier in pos-neg order
    # pos_path = '../../Data/offense.train.txt'
    # neg_path = '../../Data/other.train.txt'

    # load training data of ensemble classifier
    train_path = '../../Data/germeval.ensemble.train.txt'
    Xtrain, Ytrain = read_corpus(train_path)
    assert len(Xtrain) == len(Ytrain), 'Unequal length for Xtrain and Ytrain!'
    print('{} training samples'.format(len(Xtrain)))

    # Set up vectorizer to get the SentLen and Lexicon look-up information
    # Meta classifier uses as features the predictions of the two lower-level classifiers + SentLen + Lexicon
    print('Setting up meta_vectorizer...')
    meta_vectorizer = FeatureUnion([('length', features.TweetLength()),
                                    ('badwords', features.Lexicon('lexicon.txt'))])
    X_feats = meta_vectorizer.fit_transform(Xtrain)

    # load in predictions for training data by 1) svm, 2) cnn and 3)  multi-class svm
    # Predictions already saved as scipy sparse matrices
    print('Loading SVM, CNN predictions + Multi-class SVM predictions for train...')
    f1 = open('NEW-train-svm-predict.p', 'rb')
    SVM_train_predict = pickle.load(f1)
    f1.close()
    f2 = open('NEW-train-cnn-predict.p', 'rb')
    CNN_train_predict = pickle.load(f2)
    f2.close()
    f3 = open('Multi-train-predict.p', 'rb')
    Multi_train_predict = pickle.load(f3)
    f3.close()

    # Combine all features to input to ensemble classifier
    print('Stacking all features...')
    Xtrain_feats = hstack((X_feats, SVM_train_predict, CNN_train_predict, Multi_train_predict))
    print(type(Xtrain_feats))
    print('Shape of featurized Xtrain:', Xtrain_feats.shape)

    # Set-up meta classifier
    # meta_clf = Pipeline([('clf', LinearSVC(random_state=0))]) # LinearSVC
    meta_clf = Pipeline([('clf', LogisticRegression(random_state=0))]) # Logistic Regressor

    # Fit it
    print('Fitting meta-classifier...')
    meta_clf.fit(Xtrain_feats, Ytrain)


    '''
    PART: TESTING META-CLASSIFIER
    '''

    # load test data of ensemble classifier in pos-neg order
    # pos_path = '../../Data/offense.test.txt'
    # neg_path = '../../Data/other.test.txt'

    # load test data of ensemble classifier
    test_path = '../../Data/germeval.ensemble.test.txt'
    Xtest, Ytest = read_corpus(test_path)
    assert len(Xtest) == len(Ytest), 'Unequal length for Xtest and Ytest!'
    print('{} test samples'.format(len(Xtest)))

    Xtest_feats = meta_vectorizer.transform(Xtest)

    # Loading predictions of SVM, CNN and Multi-class SVM on test data
    print('Loading SVM, CNN predictions + Multi-class SVM predictions for test...')
    ft1 = open('NEW-test-svm-predict.p', 'rb')
    SVM_test_predict = pickle.load(ft1)
    ft1.close()
    ft2 = open('NEW-test-cnn-predict.p', 'rb')
    CNN_test_predict = pickle.load(ft2)
    ft2.close()
    f3 = open('Multi-test-predict.p', 'rb')
    Multi_test_predict = pickle.load(f3)
    f3.close()

    # Combine all features for test input to input to ensemble classifier
    print('Stacking all features...')
    Xtest_feats = hstack((Xtest_feats, SVM_test_predict, CNN_test_predict, Multi_test_predict))
    print('Shape of featurized Xtest:', Xtest_feats.shape)

    # Use trained meta-classifier to get predictions on test set
    Yguess = meta_clf.predict(Xtest_feats)

    # Evaluate
    evaluate(Ytest, Yguess)


    '''
    Results:

    Meta classifier = LinearSVC
    --------------------------------------------------
    Accuracy: 0.7325349301397206
    --------------------------------------------------
    Precision, recall and F-score per class:
                Precision     Recall    F-score
    OFFENSE      0.611732   0.629310   0.620397
    OTHER        0.799689   0.787462   0.793529
    --------------------------------------------------
    Average (macro) F-score: 0.7069625529797422
    --------------------------------------------------
    Confusion matrix:
    Labels: ['OFFENSE', 'OTHER']
    [[219 129]
     [139 515]]

    Meta classifer = Logistic Regressor
    --------------------------------------------------
    Accuracy: 0.7564870259481038
    --------------------------------------------------
    Precision, recall and F-score per class:
                Precision     Recall    F-score
    OFFENSE      0.750000   0.448276   0.561151
    OTHER        0.758186   0.920489   0.831492
    --------------------------------------------------
    Average (macro) F-score: 0.6963213959219364
    --------------------------------------------------
    Confusion matrix:
    Labels: ['OFFENSE', 'OTHER']
    [[156 192]
     [ 52 602]]



    '''
