import os
import sys
import numpy as np
import csv
import reader_full
import functions
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant


MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 100000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

embeddings_index = functions.read_word_vectors(5)

data = reader_full.read_dataset(0,1246)

texts, essayset, essaynumber, targets = functions.process_texts(data)

sequences, word_index = functions.texts_to_sequences(MAX_NUM_WORDS, texts)

MAX_SEQUENCE_LENGTH = min(MAX_SEQUENCE_LENGTH, functions.longest_text(sequences))

pad_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) #adds zeros to beginning of text if it is shorter than MAX_SEQUENCE_LENGTH

print('Shape of data tensor:', pad_sequences.shape)
print('Shape of target tensor:', targets.shape)

# split the data into a training set and a validation set
x_train, d_train, x_val, d_val = functions.split_data(pad_sequences, essayset, essaynumber, targets, VALIDATION_SPLIT)

embedding_layer = functions.embedding_layer(MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH, word_index, EMBEDDING_DIM, embeddings_index, randomize_unseen_words = True, trainable = True)


model = functions.create_model( MAX_SEQUENCE_LENGTH, embedding_layer, layers = 1, kernels = 2, kernel_length = 5)
