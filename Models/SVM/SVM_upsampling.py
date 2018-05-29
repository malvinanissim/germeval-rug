'''
SVM systems for germeval
'''
import argparse
import re
import statistics as stats
import stop_words
import json
import random

# from features import get_embeddings
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


class Embeddings(TransformerMixin):
    '''Transformer object turning a sentence (or tweet) into a single embedding vector'''

    def __init__(self, word_embeds):
        ''' Required input: word embeddings stored in dict structure available for look-up '''
        self.word_embeds = word_embeds

    def transform(self, X, **transform_params):
        '''
        Transformation function: X is list of sentence/tweet - strings in the train data. Returns list of embeddings, each embedding representing one tweet
        '''
        return [self.get_sent_embedding(sent, self.word_embeds) for sent in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_sent_embedding(self, sentence, word_embeds):
        '''
        Obtains sentence embedding representing a whole sentence / tweet
        '''
        # simply get dim of embeddings
        l_vector = len(word_embeds[':'])

        # replace each word in sentence with its embedding representation via look up in the embedding dict strcuture
        # if no word_embedding available for a word, just ignore the word
        # [[0.234234,-0.276583...][0.2343, -0.7356354, 0.123 ...][0.2344356, 0.12477...]...]
        list_of_embeddings = [word_embeds[word.lower()] for word in sentence.split() if word.lower() in word_embeds]

	    # Obtain sentence embeddings either by average or max pooling on word embeddings of the sentence
        sent_embedding = [sum(col) / float(len(col)) for col in zip(*list_of_embeddings)]  # average pooling
        # sent_embedding = [max(col) for col in zip(*list_of_embeddings)]	# max pooling

        # Below case should technically not occur
        if len(sent_embedding) != l_vector:
            sent_embedding = [0] * l_vector
        return sent_embedding



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

def cross_val(X, Y, vectorizer, clf, folds, task):
    '''Customised cross_val with chosen classifier and vectorizer'''

    kf = KFold(n_splits=folds)
    # store eval metric scores for each fold to take average over them
    precs, recs, f1s, f1_macros, accs = [],[],[],[],[]
    count = 1
    for train_idx, test_idx in kf.split(X):

        print('Working on fold %d ...' % count)

        Xtrain = X[train_idx]
        Ytrain = Y[train_idx]
        Xtest = X[test_idx]
        Ytest = Y[test_idx]
        # if task is multi, upsample the classes INSULT and PROFANITY
        # if task = 'multi':
            # new_dataset = upsample(Xtrain, Ytrain, class, factor)

        # Fitting vectorizer on training data
        Xtrain = vectorizer.fit_transform(Xtrain)
        print('Shape of Xtrain after vectorizing: ', Xtrain.shape)

        clf.fit(Xtrain, Ytrain)
        Yguess = clf.predict(vectorizer.transform(Xtest))

        # evaluate this fold
        prec, rec, f1, f1_macro, accuracy = eval_fold(Ytest, Yguess)
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

# def upsample(X_orig, Y_orig, class, factor):
#     ''' Customised function for upsampling '''
#     pass

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
    parser.add_argument('--folds', metavar='nf', type=int, default=4, help='Number of folds for cross-validation')
    parser.add_argument('--upsample', dest='upsample', action='store_true', help='Add this flag to upsample classes INSULT and PROFANITY, use only when task is \'multi\'')
    parser.set_defaults(upsample=False)
    args = parser.parse_args()

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

    # X = X[:500]
    # Y = X[:500]

    # Splitting into train and test set
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
    # print(len(Xtrain))
    # print(len(Xtest))
    # print(len(Ytrain))
    # print(len(Ytest))

    '''
    Do upsampling on training set for classes 'PROFANITY' and 'INSULT' if needed
    '''
    if args.upsample:
        UP_FACTOR = {'INSULT':3, 'PROFANITY':9}

        insult, profanity = [],[]
        for idx in range(len(Xtrain)):
            # if a sample has a class to be upsampled, store it
            if Ytrain[idx] == 'INSULT':
                insult.append(Xtrain[idx])
            elif Ytrain[idx] == 'PROFANITY':
                profanity.append(Xtrain[idx])

        # Upsampling according to prespecified factors
        # Minus 1 because if we want the num of samples with class C in training data to be X times of what it really is, we need to add X-1 copies of those samples to training data
        insult = insult * (UP_FACTOR['INSULT'] - 1)
        profanity = profanity * (UP_FACTOR['PROFANITY'] -1)

        # Adding aritificial copies to training data
        for sample in insult:
            Xtrain.append(sample)
            Ytrain.append('INSULT')
        for sample in profanity:
            Xtrain.append(sample)
            Ytrain.append('PROFANITY')

        # shuffle training data so added copies are not necessarily at the end
        temp = list(zip(Xtrain, Ytrain))
        random.shuffle(temp)
        Xtrain[:], Ytrain[:] = zip(*temp)

    print('len(Xtrain) with upsample set to {}: {}'.format(args.upsample, len(Xtrain)))
    print('len(Ytrain)', len(Ytrain))
    print('len(Xtest) with upsample set to {}: {}'.format(args.upsample, len(Xtest)))
    # print(len(Ytest))


    '''
    Actual modelling
    '''

    # Vectorizing data / Extracting features
    print('Preparing tools (vectorizer, classifier) ...')

    # unweighted word uni and bigrams
    count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('de'))
    count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))

    # Getting twitter embeddings
    # path_to_embs = '../../Resources/wiki_embeddings_de_52D.json'
    # print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
    # embeddings = json.load(open(path_to_embs, 'r'))
    # print('Done')

    vectorizer = FeatureUnion([('word', count_word),
                                ('char', count_char)])
    #                            ('word_embeds', Embeddings(embeddings))])
    # vectorizer = count_word


    # Set up SVM classifier with unbalanced class weights
    if args.task.lower() == 'binary':
        # le.transform() takes an array-like object and returns a np.array
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

    classifier = Pipeline([
                            ('vectorize', vectorizer),
                            ('classify', clf)
    ])

    # Fitting classifier
    print('Fitting model...')
    classifier.fit(Xtrain, Ytrain)

    # Predicting
    print('Predicting...')
    Yguess = classifier.predict(Xtest)

    # Evaluating
    evaluate(Ytest, Yguess)


    # print('Training and cross_validating...')
    # scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    #
    # # cross_validate takes care of fitting and predicting
    # scores = cross_validate(classifier, X, Y, scoring=scoring, cv=args.folds, return_train_score=False)
    #
    # # Outputting results
    # for metric, values in scores.items():
    #     if not metric.endswith('time'):
    #         print(metric)
    #         print(values)
    #         print('Mean scores:', stats.mean(values))
