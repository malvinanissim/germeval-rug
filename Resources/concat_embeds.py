'''
This is meant to concatenate two sets of embeddings
Embedding 1 can be json-loaded to dictionary structure
Embedding 2 is a gensim model
Padding used for vocabulary that's in one but not the other embeddings
'''
from collections import defaultdict

import numpy as np
import gensim.models as gm
import json

def save_embeddings(embeddings, outfile):
    ''' Saving embeddings to JSON object '''
    with open(outfile, 'w', encoding='utf-8') as fo:
        json.dump(embeddings, fo)

PATH_WIKI = 'test_embeddings.json'
PATH_HATE = 'model.bin'

f = open(PATH_WIKI, 'r')
wikiE = json.load(f)
f.close()

hateE = gm.KeyedVectors.load(PATH_HATE).wv
'''
wikiE: dict, key = word(str), value = embeddings(list)
hateE: gensim obj allowing look up, key = word(str), value = embeddings(nd.array, dtype=float32)

Output to be json-dumped in dict
'''
# Their respective dimensions
dim_wikiE = len(wikiE['und'])
dim_hateE = len(hateE['und'])
print('Dimensions of Wiki embeddings:', dim_wikiE)
print('Dimensions of Hate embeddings:', dim_hateE)
print('Vocab of Wiki embeddings:', len(wikiE))
print('Vocab of Hate embeddings:', len(hateE.index2word))

# Loop through Wiki entries
print('Looping through Wiki embeddings...')
joinedE = dict()
for word, emb in wikiE.items():
    # if word is also in vocab of hateE, just concatenate
    if word in hateE:
        # wemb = np.array(emb, dtype='float32')
        wemb = emb
        hemb = hateE[word]
        # wemb and hemb should now be of the same type, viz. nd.array, dtype='float32'
        # concat them and save in joinedE
        joinedE[word] = np.concatenate((wemb, hemb))
    # if word NOT in vocab of hateE, pad with average value of all embedding-values in wikiE
    else:
        pad = np.array([ np.mean(emb) for i in range(dim_hateE)], dtype='float32')
        # wemb = np.array(emb, dtype='float32')
        wemb = emb
        joinedE[word] = np.concatenate((wemb, hemb))

print('Vocab of joinedE:', len(joinedE))
print('Dimension of joinedE:', len(joinedE['und']))

# Loop through Hate entries to get words in Hate but not in Wiki
print('Looping through vocab of Hate embeddings...')
for word in hateE.index2word:
    if word not in wikiE:
        pad = np.array([ np.mean(hateE[word]) for i in range(dim_wikiE)], dtype='float32')
        joinedE[word] = np.concatenate((pad, hateE[word]))

print('Final vocab of joinedE:', len(joinedE))
print('Dimension of joinedE:', len(joinedE['und']))

'''
joinedE is an ordinary python dict object, keys are words (str), values are embeddings (nd.array, dtype='float32')
'''

# Saving new, concatenated embeddings
# save_embeddings(joinedE, 'joined_embeddings_D.json') --> does not work because nd.array are not serializable to json -> if really needed, convert to list first.



















#### space #####
