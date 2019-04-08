import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.initializers import Constant
from BenHamner.score import quadratic_weighted_kappa
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def read_word_vectors(filepath, stop = -1):

    print('Indexing word vectors.')

    embeddings_index = {}
    counter = 0
    f = open(filepath, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        counter += 1
        if counter == stop:
            break;
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


def quadratic_weighted_kappa_for_cnn(x_val, d_val, model):

    p = model.predict([x_val])
    y_test = []
    d_test = []
    for i in range(len(x_val)):
        y_test.append(np.argmax(p[i]))
        d_test.append(np.argmax(d_val[i]))

    kappa = quadratic_weighted_kappa(d_test, y_test)
    return kappa


def process_texts(data):

    print('Processing text dataset')

    texts = []  # list of text samples
    essayset  = [] #list of which set each text belongs to
    essaynumber = []
    targets = []

    for row in data:
        texts.append(row[2])
        essayset.append(int(row[1]))
        essaynumber.append(int(row[0]))
        targets.append(int(row[6])-2) #changing grades from 2-13 to 0-11

    targets = to_categorical(np.asarray(targets)) #creates a target vector for each text. If a text belongs to class 0 out of 4 classes the vector will be: [1., 0., 0., 0.]
    essayset = np.array(essayset)
    essaynumber = np.array(essaynumber)

    print('Found %s texts. ' % len(texts))

    return texts, essayset, essaynumber, targets


def texts_to_sequences(MAX_NUM_WORDS, texts):

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts) #list of all texts where words are numbers instead
    word_index = tokenizer.word_index #dictionary mapping each word to the correct number

    print('Found %s unique tokens.' % len(word_index))

    return sequences, word_index


def longest_text(texts):
    length = 0
    for text in texts:
        if length < len(text):
            length = len(text)

    return length


# split the data into a training set and a validation set
def split_data(pad_sequences, essayset, essaynumber, targets, VALIDATION_SPLIT):

    indices = np.arange(pad_sequences.shape[0]) #creates an array with integers up to the total number of texts (data.shape[0]). ex: [0  1  2  3 ... 1998  1999]
    np.random.shuffle(indices)
    pad_sequences = pad_sequences[indices]
    essayset = essayset[indices]
    essaynumber = essaynumber[indices]
    targets = targets[indices]
    num_validation_samples = int(VALIDATION_SPLIT * pad_sequences.shape[0])

    x_train = pad_sequences[:-num_validation_samples]
    d_train = targets[:-num_validation_samples]
    x_val = pad_sequences[-num_validation_samples:]
    d_val = targets[-num_validation_samples:]

    return x_train, d_train, x_val, d_val


def embedding_layer(MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH, word_index, EMBEDDING_DIM, embeddings_index, randomize_unseen_words = True, trainable = True):

#    print('Preparing embedding matrix.')
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        elif randomize_unseen_words == True:
            embedding_matrix[i] = np.random.randint(100,1000,EMBEDDING_DIM)/1000
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=trainable)

#    count = 0
#    zeros = np.zeros(100)
#    for i in range(num_words):
#            if embedding_matrix[i][0] == 0 and embedding_matrix[i][1] == 0 and embedding_matrix[i][2] == 0 and embedding_matrix[i][3] == 0:
#                count += 1
#
#    print("Unused words: " ,count, "/",  num_words)

    return embedding_layer






def create_model(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 1, kernels = 1, kernel_length = 1, dense=1, dropout=0, maxpooling = 5):

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = Dropout(dropout)(embedded_sequences)
    x = Conv1D(kernels, kernel_length, activation='relu')(embedded_sequences)
    x = MaxPooling1D(maxpooling)(x)
    x = Conv1D(kernels, 3, activation='relu')(x)
    #x = MaxPooling1D(5)(x)
    #x = Conv1D(kernels, kernel_length, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(dense, activation='sigmoid')(x)
    preds = Dense(11, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    return model

#testa 1 lager
#testa lite Dense
#fixed embedding
#rattade ord
#slumpmassiga ord
#confusion matrix!
#dropout och l2




def create_model_two(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 1, kernels = 1, kernel_length = 1):

    print('Creating model...')
    # train a 1D convnet with global maxpooling
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Embedding
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

    seq_length = 783

    model = Sequential()
    model.add(Conv1D(1, 1, activation='relu', input_shape=(seq_length,100)))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(1, 1, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(1, 1, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(11, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.summary()

    return model









def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    #
    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def argmax(x_val, d_val, model):

    p = model.predict([x_val])
    y_test = []
    d_test = []
    for i in range(len(x_val)):
        y_test.append(np.argmax(p[i]))
        d_test.append(np.argmax(d_val[i]))

    return(y_test, d_test)



def save_confusion_matrix(savefile, model, x, d, lowest_score, highest_score, title=None):

    predictions, targets = argmax(x, d, model)

    class_names = np.array(lowest_score)
    for i in range(lowest_score+1,highest_score+1):
        class_names = np.append(class_names, i)

    plot = plot_confusion_matrix(targets, predictions, classes=class_names,
                      title=title)
    plt.savefig(savefile)
