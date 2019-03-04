import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant


BASE_DIR = r"C:\Users\Edvin\Projects\Data"
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 100000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


print('Indexing word vectors.')

embeddings_index = {}
counter = 0
max_words = -1
f = open(r"C:\Users\Edvin\Projects\Data\glove.6B\glove.6B.100d.txt", encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    counter += 1
    if counter == max_words:
        break;
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedded_words = []
for word, vector in embeddings_index.items():
    embedded_words.append(word)

print(embedded_words)
