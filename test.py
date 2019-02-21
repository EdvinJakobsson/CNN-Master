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
import reader

print("done")

# BASE_DIR = r"C:\Users\Edvin\Projects\Data"
# GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
# TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
#
# MAX_SEQUENCE_LENGTH = 1000
# MAX_NUM_WORDS = 100000
# EMBEDDING_DIM = 100
# VALIDATION_SPLIT = 0.2
#
#
#
# data = reader.read_dataset(3)
#
#
#
# print('Processing text dataset')
#
# texts = []  # list of text samples
# labels_index = {}  # dictionary mapping label name to numeric id
# labels = []  # list of label ids
# targets = []
#
# for row in data:
#     texts.append(row[2])
#     labels.append(row[0])
#     targets.append(row[6])
#
#
# print('Found %s texts. ' % len(texts))
#
#
# tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts) #list of all texts where words are numbers instead
# word_index = tokenizer.word_index #dictionary mapping each word to the correct number
# print(word_index)
#
# # words = text_to_word_sequence(texts[0])
# # #print(type(word_index))
# # #print(sequences[0])
# # inv_map = {v: k for k, v in word_index.items()}
# # print(inv_map)
# # for wordnumber in sequences[0][:5]:
# #     print(inv_map[wordnumber])
#
#
# print('Found %s unique tokens.' % len(word_index))
