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

BASE_DIR = r"C:\Users\Edvin\Projects\CNN-Master"
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 100000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


print('Indexing word vectors.')

embeddings_index = {}
counter = 0
max_words = -2
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





print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

i = 0
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    if i == 2:
        break;
        i += 1
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t =f.read()
                    i = t.find('\n\n')
                    if 0 <i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)

print('Found %s texts. ' % len(texts))



tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts) #list of all texts where words are numbers instead
word_index = tokenizer.word_index #dictionary mapping each word to the correct number


print('Found %s unique tokens.' % len(word_index))


data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) #adds zeros to beginning of text if it is shorter than MAX_SEQUENCE_LENGTH
labels = to_categorical(np.asarray(labels)) #creates a target vector for each text. If a text belongs to class 0 out of 4 classes the vector will be: [1., 0., 0., 0.]
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
