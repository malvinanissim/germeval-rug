'''
This script is to be used for the ensemble system to get CNN predictions.
This CNN needs 2 datasets, train and test, training itself on the trainset and outputting predictions for each X in testset
Predictions stored in pickle
'''

import numpy as np
import pickle
import statistics as stats
# import data_helpers_final_runs
import data_helpers
from w2v_xy import train_word2vec
from scipy.sparse import csr_matrix

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from numpy import array
# import pandas as pd

np.random.seed(0)

# ---------------------- Parameters section -------------------
#

model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static

# Data source
data_source = "local_dir"  # keras_data_set|local_dir

# Model Hyperparameters
embedding_dim = 300
filter_sizes = (3, 5, 8)
num_filters = 6
dropout_prob = (0.6, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64
num_epochs = 10

# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10


#
# ---------------------- Parameters end -----------------------


def load_data(data_source):
    assert data_source in ["local_dir"]
    x, y, vocabulary, vocabulary_inv_list, len_train, idx_espresso = data_helpers.load_data() # idx_espresso is irrelevant in this case
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    y = y.argmax(axis=1)

    return x, y, len_train, vocabulary_inv


# def pad_samples_0(samples, len_needed):
#     """
#     Pads each sample in dataset with 0s to len_needed
#     Used to make samples in train and test set compatible in shape (same dimension along axis 1)
#     """
#     padded_sentences = []
#     for samp in samples:
#         num_padding = len_needed - len(samp)
#         # pad needed amount of 0 to the right (end) of the sample
#         new_sentence = np.pad(samp, (0, num_padding), 'constant', constant_values=(0, 0))
#         padded_sentences.append(new_sentence)
#     return np.array(padded_sentences)

def evaluate(Ygold, Yguess):
    '''Evaluating model performance and printing out scores in readable way'''

    print('-'*50)
    print("Accuracy:", accuracy_score(Ygold, Yguess))
    print('-'*50)
    print("Precision, recall and F-score per class:")

    # get all labels in sorted way
    # Ygold is a regular list while Yguess is a numpy array
    labs = sorted(set(Ygold + Yguess))


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


# Get full dataset
print("Load data...")
X, Y, len_train, vocabulary_inv = load_data(data_source)
# print("Xtrain shape:", Xtrain.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# Resplit into train and test data using len_train!
# Recall the full dataset is TRAIN + TEST
Xtrain, Ytrain = X[:len_train], Y[:len_train]
Xtest, Ytest = X[len_train:], Y[len_train:]

print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)
assert Xtrain.shape[1] == Xtest.shape[1]


if sequence_length != Xtrain.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = Xtrain.shape[1]

# Prepare embedding layer weights and convert inputs for static model
print("Model type is", model_type)
if model_type in ["CNN-non-static", "CNN-static"]:
    embedding_weights = train_word2vec(X, vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    # Below if-clause does not concern us since we will use CNN-non-static model
    if model_type == "CNN-static":
        x = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x])
        # x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
        print("x static shape:", x.shape)
        # print("x_test static shape:", x_test.shape)

elif model_type == "CNN-rand":
    embedding_weights = None
else:
    raise ValueError("Unknown model type")

# Build model
if model_type == "CNN-static":
    input_shape = (sequence_length, embedding_dim)
else:
    input_shape = (sequence_length,)

model_input = Input(shape=input_shape)

# Static model does not have embedding layer
if model_type == "CNN-static":
    z = model_input
else:
    z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

z = Dropout(dropout_prob[0])(z)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

assert len(Xtrain) == len(Ytrain), 'Difference in len between Xtrain and Ytrain!'
assert len(Xtest) == len(Ytest), 'Difference in len between Xtest and Ytest!'

# Finalize model building
model = Model(model_input, model_output)
# Compile model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# Initialize weights with word2vec
if model_type == "CNN-non-static":
    weights = np.array([v for v in embedding_weights.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([weights])

# Fit model
print('Fitting model...')
model.fit(Xtrain, Ytrain, epochs=num_epochs, batch_size=batch_size, verbose=2)

# Get predictions
print('Predicting...')
Ycnn = model.predict(Xtest)

Y_classes = (Ycnn > 0.5).astype(np.int)
print('classification report...')
print(classification_report(Ytest, Y_classes))
print()
# print(Y_classes[0])
# print(Y_classes[0].shape)
Yguess = []
for pred in Y_classes:
    if pred[0] == 0:
        Yguess.append('OTHER')
    elif pred[0] == 1:
        Yguess.append('OFFENSE')
    else:
        raise ValueError

Ygold = []
for pred in Ytest:
    if pred == 0:
        Ygold.append('OTHER')
    elif pred == 1:
        Ygold.append('OFFENSE')
    else:
        raise ValueError
# print(len(Ygold), 'true test labels')

evaluate(Ygold, Yguess)
# print()
# print(classification_report(Ygold, Yguess))



###
