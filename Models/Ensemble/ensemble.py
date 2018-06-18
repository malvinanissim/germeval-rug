'''
This script implements an ensemble classifer for GermEval 2018.
The lower-level classifiers are SVM and CNN
the meta-level classifer is an SVM

At test time, we import predictions made by SVM and CNN as features
'''

import argparse
import re
import statistics as stats
import stop_words
import features
import json
import pickle
from scipy.sparse import hstack

# from features import get_embeddings
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
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

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run models for either binary or multi-class task')
    parser.add_argument('file', metavar='f', type=str, help='Path to data file')
    parser.add_argument('--task', metavar='t', type=str, default='binary', help="'binary' for binary and 'multi' for multi-class task")
    parser.add_argument('--folds', metavar='nf', type=int, default=4, help='Number of folds for cross-validation in lower-level classifiers')
    args = parser.parse_args()

    '''
    Setting up the lower-level classifiers:
    SVM with word + character ngrams + Twitter embeddings
    '''

    print('Setting up lower-level SVM...')

    # Getting embeddings
    path_to_embs = '../../Resources/test_embeddings.json'
    print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
    embeddings, vocab = load_embeddings(path_to_embs)
    print('Done')

    # unweighted word uni and bigrams
    count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('de'))
    count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))

    vectorizer = FeatureUnion([('word', count_word),
                                ('char', count_char),
                                ('word_embeds', features.Embeddings(embeddings, pool='max'))])

    # Set up SVM classifier with unbalanced class weights
    if args.task.lower() == 'binary':
        cl_weights_binary = {'OTHER':1, 'OFFENSE':3}
        clf = LinearSVC(class_weight=cl_weights_binary)
    else:
        cl_weights_multi = {'OTHER':0.5,
                            'ABUSE':3,
                            'INSULT':3,
                            'PROFANITY':4}
        clf = LinearSVC(class_weight=cl_weights_multi)

    svm = Pipeline([ ('vectorize', vectorizer),
                            ('classify', clf)])

    '''
    ######################
    'CNN?'
    #####################
    '''


    print('Finished preparing lower-level classifiers')

    print('Reading in data...')
    if args.task.lower() == 'binary':
        X,Y = read_corpus(args.file)
    else:
        X,Y = read_corpus(args.file, binary=False)

    # Minimal preprocessing: Removing line breaks
    Data_X = []
    for tw in X:
        tw = re.sub(r'@\S+','User', tw)
        tw = re.sub(r'\|LBR\|', '', tw)
        tw = re.sub(r'#', '', tw)
        Data_X.append(tw)
    X = Data_X

    # Splitting into train and test set for training meta-classifier
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

    print('Getting SVM predictions on training data through cross-validation...')

    # obtain list of predictions the SVM makes for each sample in Xtrain, through cross validation
    Ytrain_svm_pred = cross_val_predict(svm, Xtrain, Ytrain, cv=args.folds)


    ###########
    print('Getting CNN predictions on training data through cross-validation...')
    # Getting list of prediction using CNN
    # Ytrain_cnn_pred =
    ###########

    '''
    training meta-classifier
    '''

    print('Training meta classifier...')
    print('Turning X into feature representation (4 features)...')

    # Meta classifier uses as features the predictions of the two lower-level classifiers + SentLen + Lexicon
    # vec_badwords = Pipeline([('badness', features.BadWords('lexicon.txt')), ('vec', DictVectorizer())])

    meta_vectorizer = FeatureUnion([('length', features.TweetLength()),
                                    ('badwords', features.Lexicon('lexicon.txt'))])

    X_feats = meta_vectorizer.fit_transform(Xtrain)

    # Transform SVM and CNN prediction into scipy matrix to make hstack possible
    Ytrain_svm_pred = csr_matrix.transpose(csr_matrix(Ytrain_svm_pred))
    Ytrain_cnn_pred = csr_matrix.transpose(csr_matrix(Ytrain_cnn_pred))

    # Get final feature representation of Xtrain for training of meta-classifier
    # Features: TweetLength, Lexicon, SVM-prediction, CNN-prediction
    Xtrain_feats = hstack((X_feats, Ytrain_svm_pred, Ytrain_cnn_pred))


    # Set-up meta classifier, a linear SVM again
    meta_clf = Pipeline([('clf', LinearSVC(random_state=0))])

    # Fit it
    print('Fitting meta-classifier...')
    meta_clf.fit(Xtrain_feats, Ytrain)


    '''
    Testing meta classifier on val set
    '''

    # Get SVM predictions of test set
    Ytest_svm_pred = svm.predict(Xtest)

    # Get CNN predictions of test set
    Ytest_cnn_pred = ??

    # Get final feature representation of test set
    X_test_feats = meta_vectorizer.transform(Xtest)
    Xtest_feats = hstack((X_test_feats, Ytest_svm_pred, Ytest_cnn_pred))

    # Final meta classifier prediction on test set
    Yguess = meta_clf.predict(Xtest_feats)

    # Evaluating final meta classifier prediction
    evaluate(Ytest, Yguess)



















####### space ########
