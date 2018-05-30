import sys
import re
import json
import pickle
import gensim.models as gm
from nltk.tokenize import word_tokenize

with open('germeval2018.training.txt', 'r', encoding='utf-8') as fi:
    content = [line.strip().split('\t')[0] for line in fi]
# print(content[:5])

# clean data
vocab = set()
for tw in content:
    tw = re.sub(r'@\S+','User', tw)
    tw = re.sub(r'\|LBR\|', '', tw)
    tw = re.sub(r'#', '', tw)
    vocab.update([ token.lower() for token in word_tokenize(tw)])

print('{} unique tokens in training data'.format(len(vocab)))

path_embed = sys.argv[1]
if path_embed.endswith('json'):
    f = open(path_embed, 'rb')
    embeds = json.load(f)
    f.close
    embed_vocab = {k.lower() for k,v in embeds.items()}
elif path_embed.endswith('bin'):
    embeds = gm.KeyedVectors.load(path_embed).wv
    embed_vocab = {word.lower() for word in embeds.index2word}
elif path_embed.endswith('p'):
    f = open(path_embed,'rb')
    embeds = pickle.load(f)
    f.close
    embed_vocab = {k.lower() for k,v in embeds.items()}


print('Vocab size of embeddings: {}'.format(len(embed_vocab)))

###### Checking coverage of embeddings w.r.t. our training data #########
Ivocab = vocab.intersection(embed_vocab)
OOV = vocab - Ivocab
frac_in = round(len(Ivocab)/len(vocab),2)
frac_out = round(len(OOV)/len(vocab),2)

print('''Coverage of {}:
Words in embeddings: {}, {} of total training data
Words not in embeddings (OOV): {}, {} of total training data'''.format(path_embed, len(Ivocab), frac_in, len(OOV), frac_out))
















## space ##
