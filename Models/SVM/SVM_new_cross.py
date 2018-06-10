'''
SVM systems for germeval
'''
import argparse
import re
import statistics as stats
import stop_words
import json

# import file containing our extra features (features.py)
import features
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate
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


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run models for either binary or multi-class task')
    parser.add_argument('file', metavar='f', type=str, help='Path to data file')
    parser.add_argument('--task', metavar='t', type=str, default='binary', help="'binary' for binary and 'multi' for multi-class task")
    parser.add_argument('--folds', metavar='nf', type=int, default=4, help='Number of folds for cross-validation')
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


    # Vectorizing data / Extracting features
    print('Preparing tools (vectorizer, classifier) ...')

    # unweighted word uni and bigrams
    count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('de'))
    count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))
    # vec_badwords = Pipeline([('badness', features.BadWords('lexicon.txt')), ('vec', DictVectorizer())])
    #
    # Getting embeddings
    path_to_embs = 'PATH TO EMBEDDINGS'
    print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
    embeddings = json.load(open(path_to_embs, 'r'))
    print('Done')

    vectorizer = FeatureUnion([('word', count_word),
                                ('char', count_char),
    #                            ('badwords', vec_badwords),
    #                            ('tweetlen', features.TweetLength()),
                                ('word_embeds', features.Embeddings(embeddings, pool='max'))])

    # vectorizer = features.Lexicon('lexicon.txt')


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

    # n-fold cross-validation with selection of metrics
    # Using the tools created!
    # print('Actual training and cross-validating...')
    # In the process of which classifier and label encoders are fitted
    # precision, recall, F1, F1_macro, Accuracy = cross_val(X, Y, vectorizer, clf, args.folds, args.task)
    #
    # # Outputting results
    # labels = sorted(set(Y))
    # output(labels, precision, recall, F1, F1_macro, Accuracy)



    print('Training and cross_validating...')

    # classifier = Pipeline([('vec', TfidfVectorizer()),
    #                             ('classify', SVC(kernel=Kernel, C=C_val))])
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    # cross_validate takes care of fitting and predicting
    scores = cross_validate(classifier, X, Y, scoring=scoring, cv=args.folds, return_train_score=False)

    # print('scores type', type(scores))

    for k,v in scores.items():
        if not k.endswith('time'):
            print(k)
            print(round(stats.mean(v), 3))
