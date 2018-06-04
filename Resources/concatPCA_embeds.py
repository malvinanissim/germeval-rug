'''
This is meant to concatenate two sets of embeddings (Embedding 1 and Embedding 2)
Pass in both files containing the embeddings as command-line arguments.
Third command-line argument is name of the pickle file in which to store the newly created, concatenated embeddings
PCA dimensionality reduction used to ensure that resulting joined embeddings have exactly the dimension of the smaller of the two embddings. No padding needed
'''
from collections import defaultdict
from sklearn.decomposition import PCA

import numpy as np
import gensim.models as gm
import json
import pickle
import sys

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



PATH_E1 = sys.argv[1]
PATH_E2 = sys.argv[2]

E1, vocab_E1 = load_embeddings(PATH_E1)
E2, vocab_E2 = load_embeddings(PATH_E2)

# Their respective dimensions
# Concatenated embeddings will end up having the dimensions of the low-dim. input embeddings
dim_E1 = len(E1['und'])
dim_E2 = len(E2['und'])
dim_wanted = min(dim_E1, dim_E2)

print('Dimensions of E1 embeddings:', dim_E1)
print('Dimensions of E2 embeddings:', dim_E2)
print('Dimensions wanted:', dim_wanted)

# Determining mutual vocab and vocab in only the one or the other set of embeddings
vocab_mutual = vocab_E1.intersection(vocab_E2)
vocab_E1_only = vocab_E1.difference(vocab_E2)
vocab_E2_only = vocab_E2.difference(vocab_E1)
print('{} words in both embeddings, {} only in E1 embeddings, {} only in E2 embeddings\nTotal of {} unique words'.format(len(vocab_mutual), len(vocab_E1_only), len(vocab_E2_only), len(vocab_mutual) + len(vocab_E2_only) + len(vocab_E1_only)))

''' Matrix for mutual vocab (M1)'''
print('Working on mutual vocab...')
# get concatenated embed of mutual vocab
# tokens and embeds contain the mutual words with their respective concatenated embeds, must match in index
words, embeddings = [],[]
for token in vocab_mutual:
    words.append(token)
    embeddings.append(np.concatenate((E2[token], E1[token])))
print('len(words)', len(words))
print('len(embeddings)', len(embeddings))
# print('type(embeddings)', type(embeddings))
# print('type(embeddings[3])', type(embeddings[3]))
# print('len(embeddings[3])', len(embeddings[3]))

# join all embeddings (currently list of arrays) into single matrix feedable to sklearn's PCA
M1_unreduced = np.array(embeddings)
print('Shape of ORIG. matrix for all mutual vocab:', M1_unreduced.shape)
# Use PCA to reduce the dimension required
pca = PCA(n_components=dim_wanted)
M1_red = pca.fit_transform(M1_unreduced)
print('Shape of REDUCED matrix for all mutual vocab:', M1_red.shape)

# associate reduced-dim embeddings with the words they represent, store in dict
M1 = { words[idx]:M1_red[idx] for idx in range(len(words))}
# print(len(M1))
# print(M1['und'])
# print(len(M1['und']))

'''Matrix for words appearing only in E2 embeddings (M2)'''
print('Working on vocab only in E2 embeddings...')
if dim_E2 > dim_wanted:
    print('PCA dim. reduction needed')
    # if E2 is the one with higher dimensions, embeddings for these words need PCA
    words, embeddings = [],[]
    for token in vocab_E2_only:
        words.append(token)
        embeddings.append(E2[token])
    print('len(words)', len(words))
    print('len(embeddings)', len(embeddings))
    M2_unreduced = np.array(embeddings)
    print('Shape of ORIG. matrix:', M2_unreduced.shape)
    # Same PCA (with same components used, but refitted to M2_unreduced)
    M2_red = pca.fit_transform(M2_unreduced)
    print('Shape of REDUCED matrix:', M2_red.shape)
    # associate reduced-dim embeddings with corresponding words
    M2 = { words[idx]:M2_red[idx] for idx in range(len(words))}
else:
    print('No need for PCA')
    # if E2 is the one with lower dimensions, its dimensions will be dim_wanted. Hence do nothing to them
    M2 = dict()
    for token in vocab_E2_only:
        M2[token] = E2[token]


'''Matrix for words appearing only in E1 embeddings (M3)'''
print('Working on vocab only in E1 embeddings...')
if dim_E1 > dim_wanted:
    print('PCA dim. reduction needed')
    # if E1 is the one with higher dimensions, embeddings for these words need PCA
    words, embeddings = [],[]
    for token in vocab_E1_only:
        words.append(token)
        embeddings.append(E1[token])
    print('len(words)', len(words))
    print('len(embeddings)', len(embeddings))
    M3_unreduced = np.array(embeddings)
    print('Shape of ORIG. matrix:', M3_unreduced.shape)
    # Same PCA (with same components used, but refitted to M3_unreduced)
    M3_red = pca.fit_transform(M3_unreduced)
    print('Shape of REDUCED matrix:', M3_red.shape)
    # associate reduced-dim embeddings with corresponding words
    M3 = { words[idx]:M3_red[idx] for idx in range(len(words))}
else:
    print('No need for PCA')
    # if E1 is the one with lower dimensions, its dimensions will be dim_wanted. Hence do nothing to them
    M3 = dict()
    for token in vocab_E1_only:
        M3[token] = E1[token]

'''
Join all three look-up ables (dictionaries). There should be no overlapping between the keys (tokens) of the three dicts
'''

NEW_EMBEDDINGS = dict()
NEW_EMBEDDINGS.update(M1)
NEW_EMBEDDINGS.update(M2)
NEW_EMBEDDINGS.update(M3)

print('len/vocab size of final embedding dict:', len(NEW_EMBEDDINGS))
# Double-checking that they do all have the dimensions required
for k,v in NEW_EMBEDDINGS.items():
    if len(v) != dim_wanted:
        raise ValueError
print('done')

## Saving NEW_EMBEDDINGS
PATH_OUT = sys.argv[3]

fout = open(PATH_OUT,'wb')
pickle.dump(NEW_EMBEDDINGS, fout)
fout.close()















#### space #####
