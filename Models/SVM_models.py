'''
SVM systems for germeval
'''
import argparse
import re
import statistics as stats

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
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

def cross_val(X, Y, clf, folds):
    '''Customised cross_val with chosen classifier and vectorized input data (X) and labels as numeric labels'''

    kf = KFold(n_splits=folds)
    # store eval metric scores for each fold to take average over them
    precs, recs, f1s, f1_macros, accs = [],[],[],[],[]
    count = 1
    for train_idx, test_idx in kf.split(X):

        print('Working on fold %d ...' % count)

        clf.fit(X[train_idx], Y[train_idx])
        Yguess = clf.predict(X[test_idx])

        # evaluate this fold
        prec, rec, f1, f1_macro, accuracy = eval_fold(Y[test_idx], Yguess)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        f1_macros.append(f1_macro)
        accs.append(accuracy)
        print('Acc:', accuracy)
        print('Average F1:', f1_macro)

        count += 1

    # Obtain averages for each metric (and each class) across the n folds.
    # precs, recs and f1s are each a list of lists, f1_macros and accs are simple lists.

    precision = take_average(precs)# list
    recall = take_average(recs)# list
    F1 = take_average(f1s)# list

    F1_macro = stats.mean(f1_macros)# single val
    Accuracy = stats.mean(accs)# single val

    return precision, recall, F1, F1_macro, Accuracy

def eval_fold(Ygold, Yguess):
    ''' Evalutes performance of a single fold '''

    PRFS = precision_recall_fscore_support(Ygold, Yguess)
    prec = PRFS[0]# list
    rec = PRFS[1]# list
    f1 = PRFS[2]# list

    f1_macro = stats.mean(PRFS[2])# f1 across all classes, single val
    accuracy = accuracy_score(Ygold, Yguess)# single val

    return prec, rec, f1, f1_macro, accuracy

def take_average(metric_list):
    ''' Computes metrics for each class as averaged across all folds. Takes a list of lists '''

    n_folds = len(metric_list)
    n_classes = len(metric_list[0])

    metric_averaged = []
    for class_idx in range(n_classes):
        single_class = []
        for fold_idx in range(n_folds):
            single_class.append(metric_list[fold_idx][class_idx])
        single_class_average = stats.mean(single_class)
        metric_averaged.append(single_class_average)

    # return list with metrics for each class, as averaged across all folds
    return metric_averaged

def output(sorted_labs, prec, rec, f1, f1_macro, acc):
    ''' Outputs all eval metrics in readable way '''

    print('-'*50)
    print("Precision, Recall and F-score per class, averaged across all folds:")
    print('{:10s} {:>10s} {:>10s} {:>10s}'.format("", "Precision", "Recall", "F-score"))
    for idx, label in enumerate(sorted_labs):
        print("{0:10s} {1:10f} {2:10f} {3:10f}".format(label, prec[idx], rec[idx], f1[idx]))
    print('-'*50)
    print("Accuracy (across all folds):", acc)
    print('-'*50)
    print("F-score (macro) (across all folds):", f1_macro)
    print('-'*50)


# def evaluate(Ygold, Yguess):
#     '''Evaluating model performance and printing out scores in readable way'''
#
#     print('-'*50)
#     print("Accuracy:", accuracy_score(Ygold, Yguess))
#     print('-'*50)
#     print("Precision, recall and F-score per class:")
#
#     # get all labels in sorted way
#     # Ygold is a regular list while Yguess is a numpy array
#     labs = sorted(set(Ygold + Yguess.tolist()))
#
#
#     # printing out precision, recall, f-score for each class in easily readable way
#     PRFS = precision_recall_fscore_support(Ygold, Yguess, labels=labs)
#     print('{:10s} {:>10s} {:>10s} {:>10s}'.format("", "Precision", "Recall", "F-score"))
#     for idx, label in enumerate(labs):
#         print("{0:10s} {1:10f} {2:10f} {3:10f}".format(label, PRFS[0][idx],PRFS[1][idx],PRFS[2][idx]))
#
#     print('-'*50)
#     print("Average (macro) F-score:", stats.mean(PRFS[2]))
#     print('-'*50)
#     print('Confusion matrix:')
#     print('Labels:', labs)
#     print(confusion_matrix(Ygold, Yguess, labels=labs))
#     print()
#


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run models for either binary or multi-class task')
    parser.add_argument('file', metavar='f', type=str, help='Path to data file')
    parser.add_argument('--task', metavar='t', type=str, default='binary', help="'binary' for binary and 'multi' for multi-class task")
    parser.add_argument('--folds', metavar='nf', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()

    #### Hyper-parameters:
    # SVM
    C_val = 1
    Kernel = 'linear'

    print('Reading in data...')
    if args.task.lower() == 'binary':
        X,Y = read_corpus(args.file)
    else:
        X,Y = read_corpus(args.file, binary=False)

    # Minimal preprocessing: Removing line breaks
    Data_X = []
    for tw in X:
        #tw = re.sub(r'@\S+','User', tw)
        tw = re.sub(r'\s\|LBR\|', '', tw)
        Data_X.append(tw)


    # Vectorizing data
    print('Vectorizing data...')
    # unweighted word uni and bigrams
    count_word = CountVectorizer(ngram_range=(1,2))
    count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))
    vectorizer = FeatureUnion([('word', count_word),
                                ('char', count_char)])
    X = vectorizer.fit_transform(Data_X)
    print(X.shape)

    # numerifying labels
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    print(Y.shape)

    # Set up classifier
    clf = SVC(kernel=Kernel, C=C_val)

    # n-fold cross-validation with selection of metrics
    print('Training and cross-validating...')
    precision, recall, F1, F1_macro, Accuracy = cross_val(X, Y, clf, args.folds)

    # Outputting results
    labels = le.inverse_transform(sorted(set(Y))).tolist()
    output(labels, precision, recall, F1, F1_macro, Accuracy)


    #
    # print('Training and cross_validating...')
    #
    # classifier = Pipeline([('vec', TfidfVectorizer()),
    #                             ('classify', SVC(kernel=Kernel, C=C_val))])
    # scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    #
    # # cross_validate takes care of fitting and predicting
    # scores = cross_validate(classifier, X, Y, scoring=scoring, cv=5, return_train_score=False)
    #
    # print('scores type', type(scores))
    #
    # for k,v in scores.items():
    #     print(k)
    #     print(v)
    #
