import numpy as np
import re
import itertools
from collections import Counter
import pickle

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf

This script assumes that we have a fixed train and a fixed test set.
The data loading method assumes that the test set is the real test set and does not come with labels

"""


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r'@\S+','User', string)
    string = re.sub(r'\|LBR\|', '', string)
    string = re.sub(r'#', '', string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loading both the train and the  test set.
    Adding the espresso dataset to train
    """
    # Load train data from files
    samples, labels = [],[]
    with open('../../Data/germeval2018.training.txt','r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # get sample
            samples.append(data[0])
            # get label
            if data[1] == 'OFFENSE':
                labels.append([0,1]) # label of positive sample
            elif data[1] == 'OTHER':
                labels.append([1,0]) # label of negative sample
            else:
                raise ValueError('Unknown label!')

    # Adding Espresso data
    samples, labels, idx_espresso = add_espresso_data(samples, labels)

    # Clean and split samples
    Xtrain = [clean_str(sample) for sample in samples]
    Xtrain = [s.split(" ") for s in Xtrain] # each sample as list of words/strings
    Ytrain = np.array(labels)
    len_train = len(Xtrain)
    # We need to remember the len of train, we will put train + test together to build the vocab. Then we will recognise the first len_train items as coming from the train set

    # Load test data, no labels
    Xtest = []
    with open('../../Data/germeval2018.test.txt','r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # get sample
            Xtest.append(data[0])

    Xtest = [clean_str(sample) for sample in Xtest]
    Xtest = [s.split(" ") for s in Xtest] # each sample as list of words/strings
    # Ytest = np.array(Ytest)

    return [Xtrain, Ytrain, Xtest, len_train, idx_espresso]

def add_espresso_data(Xorig, Yorig):
    """
    Loads the espresso dataset, randomly inserts them into orig_dataset
    Order of samples in orig_dataset is preserved.
    Returns: a) dataset (X, Y) extended with  espresso data, b) indices of espresso data samples in new dataset
    (later on predictions for espresso items can be removed via these indices)
    """
    # load espresso data
    f = open('../../Data/ger-espresso-offense-only.p', 'rb')
    espresso = pickle.load(f)
    f.close()
    Xespresso = espresso[0]
    Yespresso = [[0,1] for i in range(len(Xespresso))] # We use hate-only espresso data

    # select 'marker items' in original dataset, espresso data will be put in front of each of these marker items.
    markers = np.random.choice(Xorig, size=len(Xespresso), replace=False)
    # Get indices of these marker items in orig_da, this is where to insert espresso items.
    ind2insert = np.in1d(Xorig, markers).nonzero()[0]
    # Insert espresso samples into Xorig and Yorig
    Xorig = np.array(Xorig, dtype='object') # make Xorig and Yorig arrays first
    Xnew = np.insert(Xorig, ind2insert, Xespresso)
    Yorig = np.array(Yorig)
    Ynew = np.insert(Yorig, ind2insert, Yespresso, axis=0)
    # Get indices of marker items in Ynew
    indmarkers = np.in1d(Xnew, markers).nonzero()[0]
    # Espresso items are always exactly 1 before markers in the sample array, hence their indices are:
    ind_espresso = [ind-1 for ind in indmarkers]

    return Xnew, Ynew, ind_espresso



def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    Xtrain, Ytrain, Xtest, len_train, idx_espresso = load_data_and_labels()
    # sentences, labels, idx_espresso = load_data_and_labels()
    # Vocab needs to be build on the basis of the whole dataset, so we put train and test together! TRAIN, then TEST
    X = Xtrain + Xtest # X is list while Y is np.array
    Y = Ytrain

    sentences_padded = pad_sentences(X)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    X, Y = build_input_data(sentences_padded, Y, vocabulary)
    return [X, Y, vocabulary, vocabulary_inv, len_train, idx_espresso]



def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
    yield shuffled_data[start_index:end_index]
