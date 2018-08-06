'''
Baseline systems for Germeval 2018 Shared Task
1) Most freq. class
2) Tfidf-weighted (word) unigram + linear SVM

Input corpus file as command-line argument
'''
import argparse
import statistics as stats

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

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


    train_dat = '../../Data/germeval.ensemble.train.txt'
    test_dat = '../../Data/germeval.ensemble.test.txt'

    print('Loading data...')

    Xtrain, Ytrain = read_corpus(train_dat, binary=False)
    Xtest, Ytest = read_corpus(test_dat, binary=False)

    # Setting up most frequent class (mfc) baseline
    classifer_mfc = Pipeline([('vec', CountVectorizer()),
                                ('classify', DummyClassifier(strategy='most_frequent', random_state=0))])

    # Setting up SVM baseline
    # classifier_svm = Pipeline([('vec', TfidfVectorizer()),
    #                             ('classify', SVC(kernel=Kernel, C=C_val))])

    # Fitting either model
    print('Fitting models...')
    classifer_mfc.fit(Xtrain,Ytrain)
    # classifier_svm.fit(Xtrain,Ytrain)

    # Predicting using either model
    print('Predicting...')
    Yguess_mfc = classifer_mfc.predict(Xtest)
    # Yguess_svm = classifier_svm.predict(Xtest)

    # Evaluating the performance of either model on validation data
    print('Results for most frequent class baseline:')
    evaluate(Ytest, Yguess_mfc)
    print()

    # print('Results for svm baseline:')
    # evaluate(Ytest, Yguess_svm)
