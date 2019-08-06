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
GLOVE_DIR = os.path.join(BASE_DIR, "glove.6B")
TEXT_DATA_DIR = os.path.join(BASE_DIR, "20_newsgroup")

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 100000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


print("Indexing word vectors.")

embeddings_index = {}
counter = 0
max_words = -2
f = open(r"C:\Users\Edvin\Projects\Data\glove.6B\glove.6B.100d.txt", encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype="float32")
    embeddings_index[word] = coefs
    counter += 1
    if counter == max_words:
        break
f.close()

print("Found %s word vectors." % len(embeddings_index))


print("Processing text dataset")

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

i = 0
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    if i == -2:
        break
        i += 1
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {"encoding": "latin-1"}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find("\n\n")
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)

print("Found %s texts. " % len(texts))


tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(
    texts
)  # list of all texts where words are numbers instead
word_index = tokenizer.word_index  # dictionary mapping each word to the correct number

# words = text_to_word_sequence(texts[0])
# print(type(word_index))
# print(sequences[0])
# inv_map = {v: k for k, v in word_index.items()}
# print(inv_map)
# for wordnumber in sequences[0][:5]:
#    print(inv_map[wordnumber])


print("Found %s unique tokens." % len(word_index))


data = pad_sequences(
    sequences, maxlen=MAX_SEQUENCE_LENGTH
)  # adds zeros to beginning of text if it is shorter than MAX_SEQUENCE_LENGTH
labels = to_categorical(
    np.asarray(labels)
)  # creates a target vector for each text. If a text belongs to class 0 out of 4 classes the vector will be: [1., 0., 0., 0.]
print("Shape of data tensor:", data.shape)
print("Shape of label tensor:", labels.shape)

# split the data into a training set and a validation set
indices = np.arange(
    data.shape[0]
)  # creates an array with integers up to the total number of texts (data.shape[0]). ex: [0  1  2  3 ... 1998  1999]
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


print("Preparing embedding matrix.")

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
print("number of words: ", num_words)

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    embeddings_initializer=Constant(embedding_matrix),
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False,
)


print("Training model.")

count = 0
zeros = np.zeros(100)
for i in range(num_words):
    if (
        embedding_matrix[i][0] == 0
        and embedding_matrix[i][1] == 0
        and embedding_matrix[i][2] == 0
        and embedding_matrix[i][3] == 0
    ):
        count += 1

print("unused words and total words: ", count, num_words)


# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation="relu")(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation="relu")(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation="relu")(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation="relu")(x)
preds = Dense(len(labels_index), activation="softmax")(x)

model = Model(sequence_input, preds)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])

model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=10,
    verbose=2,
    validation_data=(x_val, y_val),
)

val_loss, val_acc = model.evaluate(x_train, y_train, verbose=2)
print("training loss and acc: ", val_loss, val_acc)
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=2)
print("validation loss and acc: ", val_loss, val_acc)
#
# #model.save("kerasmodel")

count = 0
zeros = np.zeros(100)
for i in range(num_words):
    if (
        embedding_matrix[i][0] == 0
        and embedding_matrix[i][1] == 0
        and embedding_matrix[i][2] == 0
        and embedding_matrix[i][3] == 0
    ):
        count += 1

print("unused words and total words: ", count, num_words)
