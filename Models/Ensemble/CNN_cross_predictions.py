'''
This script is to be used for the ensemble system and retrieves predictions for a given set of samples by the CNN.
Espresso data refers to the PSP dataset.

This CNN needs 1 dataset, using cross validation to output a value for each X
predictions stored in pickle
'''

import numpy as np
import pickle
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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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
    x, y, vocabulary, vocabulary_inv_list, idx_espresso = data_helpers.load_data()
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    y = y.argmax(axis=1)

    return x, y, vocabulary_inv, idx_espresso


# Data Preparation
print("Load data...")
x, y, vocabulary_inv, idx_espresso = load_data(data_source)
print(len(idx_espresso), 'samples from Espresso dataset, predictions for which to be removed at the end')


if sequence_length != x.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x.shape[1]

print("x shape:", x.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# Prepare embedding layer weights and convert inputs for static model
print("Model type is", model_type)
if model_type in ["CNN-non-static", "CNN-static"]:
    embedding_weights = train_word2vec(x, vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
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


# split into input (X) and output (Y) variables
# X = np.concatenate((x_train, x_test), axis=0)
# Y = np.concatenate((y_train, y_test), axis=0)
X = x
Y = y
assert len(X) == len(Y), 'Difference in len between X and Y!'

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
# cvpreds meant to store the prediction values provided for each sample in X via cross_val, created as dummy np.array with 0.0
cvpreds = np.array([ 0.0 for i in range(len(X))])
# print('cvpreds shape:', cvpreds.shape)
cvscores = []
count = 1
for train, test in kfold.split(X, Y):
    print('Working on fold {}...'.format(count))
    print(X[train].shape)
    print(X[test].shape)

    # Setting up new, fully untrained model at each fold
    # create model
    model = Model(model_input, model_output)
    # Compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # Initialize weights with word2vec
    if model_type == "CNN-non-static":
        weights = np.array([v for v in embedding_weights.values()])
        print("Initializing embedding layer with word2vec weights, shape", weights.shape)
        embedding_layer = model.get_layer("embedding")
        embedding_layer.set_weights([weights])

	# Fit the model
    print('Fitting...')
    model.fit(X[train], Y[train], epochs=num_epochs, batch_size=batch_size, verbose=0)

	# Evaluate the model at this fold
    scores = model.evaluate(X[test], Y[test], verbose=0, batch_size=batch_size)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

    # Get predictions for X[test] at this fold and store them in cvpreds
    preds = model.predict(X[test])
    # preds is nd.array of shape (len(preds), 1), needs reshaping to (len(preds), ) to be compatible with cvpreds
    preds = preds.reshape(len(preds),)
    cvpreds[test] = preds

    # increase counter
    count += 1

print("Overall accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

print('Sanity checks on cvpreds:')
# print('type(cvpreds):', type(cvpreds))
# print('len(cvpreds):', len(cvpreds))
print('Same length as Y ?', len(cvpreds) == len(Y))
print('Any dummy 0.0 still in cvpreds?', 0.0 in cvpreds)

Yguess = cvpreds

# Removing predictions for the Espresso samples using idx_espresso
print('len(Yguess) including Espresso:', len(Yguess))
Yguess_final = np.delete(Yguess, idx_espresso)
print('len(Yguess_final) without Espresso:', len(Yguess_final))

# Turning to scipy
print('Turning to scipy:')
Ycnn = csr_matrix.transpose(csr_matrix(Yguess_final))
print(type(Ycnn))
print(Ycnn.shape)
# print(Ycnn)

# Pickling the predictions
save_to = open('NEW-train-cnn-predict.p', 'wb')
pickle.dump(Ycnn, save_to)
save_to.close()

print('Done')


