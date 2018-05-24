'''
This is meant to concatenate two sets of embeddings
Embedding 1 can be json-loaded to dictionary structure
Embedding 2 is a gensim model
PCA dimensionality reduction used to ensure that resulting joined embeddings have exactly the dimension of the smaller of the two embddings. No padding needed
'''
from collections import defaultdict
from sklearn.decomposition import PCA

import numpy as np
import gensim.models as gm
import json

# def save_embeddings(embeddings, outfile):
#     ''' Saving embeddings to JSON object '''
#     with open(outfile, 'w', encoding='utf-8') as fo:
#         json.dump(embeddings, fo)

PATH_WIKI = 'test_embeddings.json'
PATH_HATE = 'model.bin'

f = open(PATH_WIKI, 'r')
wikiE = json.load(f)
f.close()

hateE = gm.KeyedVectors.load(PATH_HATE).wv
'''
wikiE: dict, key = word(str), value = embeddings(list)
hateE: gensim obj allowing look up, key = word(str), value = embeddings(nd.array, dtype=float32)
'''
# Their respective dimensions
# Concatenated embeddings will end up having the dimensions of the low-dim. input embeddings
dim_wikiE = len(wikiE['und'])
dim_hateE = len(hateE['und'])
dim_wanted = min(dim_wikiE, dim_hateE)

print('Dimensions of Wiki embeddings:', dim_wikiE)
print('Dimensions of Hate embeddings:', dim_hateE)
print('Dimensions wanted:', dim_wanted)

# Getting vocab from both
vocab_wikiE = { word for word,emb in wikiE.items()} # if wikiE is JSON serialized dict object
# vocab_wikiE = { word for word in wikiE.index2word} # if wikiE is Gensim model
vocab_hateE = { word for word in hateE.index2word }

vocab_mututal = vocab_wikiE.intersection(vocab_hateE)
vocab_wiki_only = vocab_wikiE.difference(vocab_hateE)
vocab_hate_only = vocab_hateE.difference(vocab_wikiE)
print('{} words in both embeddings, {} only in wiki embeddings, {} only in hate embeddings\nTotal of {} unique words'.format(len(vocab_mututal), len(vocab_wiki_only), len(vocab_hate_only), len(vocab_mututal) + len(vocab_hate_only) + len(vocab_wiki_only)))

''' Matrix for mutual vocab (M1)'''
print('Working on mutual vocab...')
# get concatenated embed of mutual vocab
# tokens and embeds contain the mutual words with their respective concatenated embeds, must match in index
words, embeddings = [],[]
for token in vocab_mututal:
    words.append(token)
    embeddings.append(np.concatenate((hateE[token], wikiE[token])))
print('len(words)', len(words))
print('len(embeddings)', len(embeddings))

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

'''Matrix for words appearing only in hate embeddings (M2)'''
print('Working on vocab only in hate embeddings...')
if dim_hateE > dim_wanted:
    print('PCA dim. reduction needed')
    # if hateE is the one with higher dimensions, embeddings for these words need PCA
    words, embeddings = [],[]
    for token in vocab_hate_only:
        words.append(token)
        embeddings.append(hateE[token])
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
    # if hateE is the one with lower dimensions, its dimensions will be dim_wanted. Hence do nothing to them
    M2 = dict()
    for token in vocab_hate_only:
        M2[token] = hateE[token]


'''Matrix for words appearing only in wiki embeddings (M3)'''
print('Working on vocab only in wiki embeddings...')
if dim_wikiE > dim_wanted:
    print('PCA dim. reduction needed')
    # if wikiE is the one with higher dimensions, embeddings for these words need PCA
    words, embeddings = [],[]
    for token in vocab_wiki_only:
        words.append(token)
        embeddings.append(wikiE[token])
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
    # if wikiE is the one with lower dimensions, its dimensions will be dim_wanted. Hence do nothing to them
    M3 = dict()
    for token in vocab_wiki_only:
        M3[token] = wikiE[token]

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

## Saving NEW_EMBEDDINGS if needed















#### space #####
