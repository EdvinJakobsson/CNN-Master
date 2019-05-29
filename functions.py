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
from BenHamner.score import mean_quadratic_weighted_kappa
import reader_full
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
        if counter == stop:
            break;
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        counter += 1

    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


def quadratic_weighted_kappa_for_cnn(x_val, d_val, essayset, model, output):
    y_test, d_test = argmax(x_val, d_val, essayset, model, output)
    kappa = quadratic_weighted_kappa(d_test, y_test)

    return kappa

def quadratic_weighted_kappa_for_hybrid(x, d_val, essayset, model, output):
    y_test, d_test = argmax(x, d_val, essayset, model, output)
    kappa = quadratic_weighted_kappa(d_test, y_test)

    return kappa


def argmax(x_val, d_val, essayset, model, output):
    asap_ranges = {
    0: (0, 60),
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60)
    }
    essayset = essayset[0]
    max_score = asap_ranges[essayset][1] - asap_ranges[essayset][0]


    predictions = []
    targets = []
    if output == 'softmax':
        p = model.predict([x_val])
        for i in range(len(x_val)):
            predictions.append(np.argmax(p[i]))
            targets.append(np.argmax(d_val[i]))
    elif(output == 'sigmoid'):
        p = model.predict([x_val])
        for i in range(len(x_val)):
            predictions.append(int(p[i]*max_score+0.5))
            targets.append(int(d_val[i]*max_score))
    elif(output == 'linear'):
        p = model.predict([x_val])
        for i in range(len(x_val)):
            targets.append(int(d_val[i]))
            prediction = int(p[i]+0.5)
            if prediction > max_score:
                prediction = max_score
            if prediction < 0:
                prediction = 0
            predictions.append(prediction)
    elif(output == 'hybrid'):
        p = model.predict([x_val[0], x_val[1]])
        for i in range(len(d_val)):
            targets.append(int(d_val[i]))
            prediction = int(p[i]+0.5)
            if prediction > max_score:
                prediction = max_score
            if prediction < 0:
                prediction = 0
            predictions.append(prediction)
    else:
        print("argmax: something wrong with 'output' value")
    return(predictions, targets)




def process_texts(data, output, essays, asap_ranges, human_range):

    essayset = essays[0]
    range = asap_ranges[essayset]
    number_of_classes = range[1]-range[0]+1
    texts = []  # list of text samples
    essaysetlist  = [] #list of which set each text belongs to
    essaynumber = []
    targets = []

    for row in data:
        texts.append(row[2])
        essaysetlist.append(int(row[1]))
        essaynumber.append(int(row[0]))
        if human_range == False:
            targets.append(int(row[6])-range[0]) # -range changes grades to start at 0
        else:
            targets.append(int(row[3])-range[0]) # -range changes grades to start at 0

    if(output == 'softmax'):
        targets = to_categorical(np.asarray(targets), number_of_classes) #creates a target vector for each text. If a text belongs to class 0 out of 4 classes the vector will be: [1., 0., 0., 0.]
    elif(output == 'linear' or output == 'hybrid'):
        targets = np.array(targets)
    elif(output == 'sigmoid'):
        targets = [x / (range[1]-range[0]) for x in targets]
        targets = np.array(targets)

    essaysetlist = np.array(essaysetlist)
    essaynumber = np.array(essaynumber)
    #print('Found %s texts. ' % len(texts))

    return texts, essaysetlist, essaynumber, targets


def texts_to_sequences(MAX_NUM_WORDS, texts):

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts) #list of all texts where words are numbers instead
    word_index = tokenizer.word_index #dictionary mapping each word to the correct number

    #print('Found %s unique tokens.' % len(word_index))

    return sequences, word_index


def longest_text(texts):
    length = 0
    for text in texts:
        if length < len(text):
            length = len(text)

    return length


# split the data into a training set and a validation set
def split_data(pad_sequences, essayset, essaynumber, targets, VALIDATION_SPLIT, randomize_data):

    indices = np.arange(pad_sequences.shape[0]) #creates an array with integers up to the total number of texts (data.shape[0]). ex: [0  1  2  3 ... 1998  1999]
    if randomize_data:
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


def plot_kappa(filename, epochs, val_kappa, train_kappa = None, title = "", x_axis = ""):
    plt.plot(epochs,val_kappa, label='Validation Kappa')
    if train_kappa != None:
        plt.plot(epochs,train_kappa, "r--", label='Training Kappa')
    plt.legend()
    plt.ylabel('Kappa')
    plt.xlabel(x_axis)
    plt.ylim(-0.1,1)
    plt.title(title)

    plt.savefig(filename)
    plt.close()

def plot_loss(filename, epochs, train_loss, val_loss, title, x_axis, y_max = 2):
    plt.plot(epochs,train_loss, "r--", label='Training Loss')
    plt.plot(epochs,val_loss, label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel(x_axis)
    plt.ylim(0,y_max)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()




def human_raters_agreement():
    asap_ranges = {
    0: (0, 60),
    1: (1, 6),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 15),
    8: (0, 30)
    }
    essaysets = [[1],[2],[3],[4],[5],[6],[7],[8]]
    essayfile = "C:/Users/Edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
    human_raters_agreement = []
    for essayset in essaysets:
         set = essayset[0]
         min_rating = asap_ranges[set][0]
         max_rating = asap_ranges[set][1]
         data = reader_full.read_dataset(essayset, filepath=essayfile)
         #data = data[:int(len(data)*0.7)]    # save 30% of essays for final evaluation
         rater1 = []
         rater2 = []
         for essay in data:
             rater1.append(int(essay[3]))
             rater2.append(int(essay[4]))

         kappa = quadratic_weighted_kappa(rater1, rater2, min_rating, max_rating)
         human_raters_agreement.append(kappa)
    return human_raters_agreement


def save_confusion_matrix(savefile, x, d, model, essayset, output, asap_ranges, title=None):
    predictions, targets = argmax(x, d, essayset, model, output)

    essayset = essayset[0]
    min_score = asap_ranges[essayset][0]
    max_score = asap_ranges[essayset][1]

    class_names = np.array(min_score)
    for i in range(min_score+1, max_score+1):
        class_names = np.append(class_names, i)

    plot = plot_confusion_matrix(targets, predictions, classes=class_names, title=title)
    plt.savefig(savefile)
    plt.close()



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
