'''
SVM systems for germeval
'''
import argparse
import re
import statistics as stats
import stop_words
import json
import pickle
import gensim.models as gm

import features
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import shuffle

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

def clean_samples(samples):
    '''
    Simple cleaning: removing URLs, line breaks, abstracting away from user names etc.
    '''

    new_samples = []
    for tw in samples:
        tw = re.sub(r'@\S+','User', tw)
        tw = re.sub(r'\|LBR\|', '', tw)
        tw = re.sub(r'http\S+\s?', '', tw)
        tw = re.sub(r'\#', '', tw)
        new_samples.append(tw)

    return new_samples


# def cross_val(X, Y, clf, folds):
#     '''Customised cross_val with chosen classifier and vectorized input data (X) and labels as numeric labels'''
#
#     kf = KFold(n_splits=folds)
#     # store eval metric scores for each fold to take average over them
#     precs, recs, f1s, f1_macros, accs = [],[],[],[],[]
#     count = 1
#     for train_idx, test_idx in kf.split(X):
#
#         print('Working on fold %d ...' % count)
#
#         clf.fit(X[train_idx], Y[train_idx])
#         Yguess = clf.predict(X[test_idx])
#
#         # evaluate this fold
#         prec, rec, f1, f1_macro, accuracy = eval_fold(Y[test_idx], Yguess)
#         precs.append(prec)
#         recs.append(rec)
#         f1s.append(f1)
#         f1_macros.append(f1_macro)
#         accs.append(accuracy)
#         print('Acc:', accuracy)
#         print('Average F1:', f1_macro)
#
#         count += 1
#
#     # Obtain averages for each metric (and each class) across the n folds.
#     # precs, recs and f1s are each a list of lists, f1_macros and accs are simple lists.
#
#     precision = take_average(precs)# list
#     recall = take_average(recs)# list
#     F1 = take_average(f1s)# list
#
#     F1_macro = stats.mean(f1_macros)# single val
#     Accuracy = stats.mean(accs)# single val
#
#     return precision, recall, F1, F1_macro, Accuracy
#
# def eval_fold(Ygold, Yguess):
#     ''' Evalutes performance of a single fold '''
#
#     PRFS = precision_recall_fscore_support(Ygold, Yguess)
#     prec = PRFS[0]# list
#     rec = PRFS[1]# list
#     f1 = PRFS[2]# list
#
#     f1_macro = stats.mean(PRFS[2])# f1 across all classes, single val
#     accuracy = accuracy_score(Ygold, Yguess)# single val
#
#     return prec, rec, f1, f1_macro, accuracy
#
# def take_average(metric_list):
#     ''' Computes metrics for each class as averaged across all folds. Takes a list of lists '''
#
#     n_folds = len(metric_list)
#     n_classes = len(metric_list[0])
#
#     metric_averaged = []
#     for class_idx in range(n_classes):
#         single_class = []
#         for fold_idx in range(n_folds):
#             single_class.append(metric_list[fold_idx][class_idx])
#         single_class_average = stats.mean(single_class)
#         metric_averaged.append(single_class_average)
#
#     # return list with metrics for each class, as averaged across all folds
#     return metric_averaged
#
# def output(sorted_labs, prec, rec, f1, f1_macro, acc):
#     ''' Outputs all eval metrics in readable way '''
#
#     print('-'*50)
#     print("Precision, Recall and F-score per class, averaged across all folds:")
#     print('{:10s} {:>10s} {:>10s} {:>10s}'.format("", "Precision", "Recall", "F-score"))
#     for idx, label in enumerate(sorted_labs):
#         print("{0:10s} {1:10f} {2:10f} {3:10f}".format(label, prec[idx], rec[idx], f1[idx]))
#     print('-'*50)
#     print("Accuracy (across all folds):", acc)
#     print('-'*50)
#     print("F-score (macro) (across all folds):", f1_macro)
#     print('-'*50)



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
    # parser = argparse.ArgumentParser(description='Run models for either binary or multi-class task')
    # parser.add_argument('file', metavar='f', type=str, help='Path to data file')
    # parser.add_argument('--task', metavar='t', type=str, default='binary', help="'binary' for binary and 'multi' for multi-class task")
    # parser.add_argument('--folds', metavar='nf', type=int, default=4, help='Number of folds for cross-validation')
    # args = parser.parse_args()

    TASK = 'binary'
    # TASK = 'multi'
    GermevalData = '../../Data/germeval2018.training.txt'
    EspressoDataAll = '../../Data/ger-espresso-data.p'
    EspressoDataOffense = '../../Data/ger-espresso-offense-only.p'

    print('Reading in Germeval data...')
    if TASK == 'binary':
        X,Y = read_corpus(GermevalData)
    else:
        X,Y = read_corpus(GermevalData, binary=False)

    print('Reading in espresso (all) data...')
    f = open(EspressoDataAll, 'rb')
    espressoAll = pickle.load(f)
    f.close()
    XesAll, YesAll = espressoAll[0], espressoAll[1]
    # print(XesAll[:3])
    # print(YesAll[:3])

    print('Reading in espresso (offense only) data...')
    f = open(EspressoDataOffense, 'rb')
    espressoOffense = pickle.load(f)
    f.close()
    XesOffense, YesOffense = espressoOffense[0], espressoOffense[1]
    # print(XesOffense[:3])
    # print(YesOffense[:3])

    # Minimal preprocessing / cleaning
    X = clean_samples(X)
    XesAll = clean_samples(XesAll)
    XesOffense = clean_samples(XesOffense)

    # Take last 1500 samples from GermevalData to be test set
    # Espresso data is only interesting as training data
    Xtrain_germeval = X[:-1250]
    Ytrain_germeval = Y[:-1250]

    Xtest = X[-1250:]
    Ytest = Y[-1250:]
    print(len(Xtest), 'test samples!')

    # preparing extended data where train is germeval train + all espresso data + shuffling
    Xtrain_all = Xtrain_germeval + XesAll
    Ytrain_all = Ytrain_germeval + YesAll
    Xtrain_all, Ytrain_all = shuffle(Xtrain_all, Ytrain_all)

    # preparing extended data where train is germeval train + offense-labelled espresso data + shuffling
    Xtrain_offense = Xtrain_germeval + XesOffense
    Ytrain_offense = Ytrain_germeval + YesOffense
    Xtrain_offense, Ytrain_offense = shuffle(Xtrain_offense, Ytrain_offense)


    '''
    Preparing vectorizer and classifier
    '''

    # Vectorizing data / Extracting features
    print('Preparing tools (vectorizer, classifier) ...')

    # unweighted word uni and bigrams
    count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('de'))
    count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))

    # Getting embeddings
    # Insert path to embeddings file (json, pickle or gensim models)
    path_to_embs = '../../Resources/test_embeddings.json'
    print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
    embeddings, vocab = load_embeddings(path_to_embs)
    print('Done')

    vectorizer = FeatureUnion([('word', count_word),
                                ('char', count_char),
                                ('word_embeds', features.Embeddings(embeddings, pool='max'))])


    # Set up SVM classifier with unbalanced class weights
    if TASK == 'binary':
        # cl_weights_binary = None
        cl_weights_binary = {'OTHER':1, 'OFFENSE':3}
        clf = LinearSVC(class_weight=cl_weights_binary)
    else:
        # cl_weights_multi = None
        cl_weights_multi = {'OTHER':0.5,
                            'ABUSE':3,
                            'INSULT':3,
                            'PROFANITY':4}
        clf = LinearSVC(class_weight=cl_weights_multi)

    # train this on germeval data only
    classifier = Pipeline([
                            ('vectorize', vectorizer),
                            ('classify', clf)])


    '''
    Actual training and predicting:
    Case 1: train on Germeval data only
    Case 2: train on Germeval + all espresso data
    Case 3: train on Germeval + espresso data of class 'offense'
    '''
    print('Using germeval data only...')
    print(len(Xtrain_germeval), 'training samples')
    classifier.fit(Xtrain_germeval, Ytrain_germeval)
    Yguess1 = classifier.predict(Xtest)
    evaluate(Ytest, Yguess1)

    print('Using all espresso data...')
    print(len(Xtrain_all), 'training samples')
    classifier.fit(Xtrain_all, Ytrain_all)
    Yguess2 = classifier.predict(Xtest)
    evaluate(Ytest, Yguess2)

    print('Using only offense-labelled espresso data...')
    print(len(Xtrain_offense), 'training samples')
    classifier.fit(Xtrain_offense, Ytrain_offense)
    Yguess3 = classifier.predict(Xtest)
    evaluate(Ytest, Yguess3)




    # print('Training germeval only model')
    # print(len(Xtrain_germeval), 'train samples!')
    # classifier_germeval.fit(Xtrain_germeval, Ytrain_germeval)
    #
    # print('Training espresso model')
    # print(len(Xtrain_all), 'train samples!')
    # classifier_espresso.fit(Xtrain_all, Ytrain_all)
    #
    # # Predicting
    # print('Predicting...')
    # Yguess_germeval = classifier_germeval.predict(Xtest)
    # Yguess_espresso = classifier_espresso.predict(Xtest)
    #
    # # Evaluate
    # print('Results with training only on Germeval data:')
    # evaluate(Ytest, Yguess_germeval)
    # print()
    # print('Results with training on Germeval + Espresso data:')
    # evaluate(Ytest, Yguess_espresso)
