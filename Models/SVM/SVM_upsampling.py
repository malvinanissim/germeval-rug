'''
SVM systems for germeval
'''
import argparse
import re
import statistics as stats
import stop_words
import json
import random
import features

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
    acc = accuracy_score(Ygold, Yguess)
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
    f1 = stats.mean(PRFS[2])
    print("Average (macro) F-score:", stats.mean(PRFS[2]))
    print('-'*50)
    print('Confusion matrix:')
    print('Labels:', labs)
    print(confusion_matrix(Ygold, Yguess, labels=labs))
    print()

    return acc, f1



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

    '''
    Preparing vectorizer + classifier to be used
    '''

    # Vectorizing data / Extracting features
    print('Preparing tools (vectorizer, classifier) ...')

    # unweighted word uni and bigrams
    count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('de'))
    count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))

    # Getting twitter embeddings
    path_to_embs = ''
    print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
    embeddings, vocab = load_embeddings(path_to_embs)
    print('Done')

    vectorizer = FeatureUnion([('word', count_word),
                                ('char', count_char),
                                ('word_embeds', features.Embeddings(embeddings))])
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


    '''
    Perform N folds of training and predicting where N = args.folds
    '''

    accuracies, f1s = [],[]
    for fold in range(args.folds):
        print('This is fold number {}'.format(fold))

        # Splitting into train and test set
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

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


        # Fitting classifier
        print('Fitting model...')
        classifier.fit(Xtrain, Ytrain)

        # Predicting
        print('Predicting...')
        Yguess = classifier.predict(Xtest)

        # Evaluating
        acc, f1 = evaluate(Ytest, Yguess)
        accuracies.append(acc)
        f1s.append(f1)
        print()

    print('Results across {} folds:'.format(args.folds))
    print('Accuracy:', stats.mean(accuracies))
    print('F1 (Macro-average):', stats.mean(f1s))




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
