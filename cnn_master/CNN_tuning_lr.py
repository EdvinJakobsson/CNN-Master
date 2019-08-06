import os
import sys
import numpy as np
import csv
import reader_full
import functions
import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # removes some of the tf warnings


MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 100000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
number_of_word_embeddings = -1  # all of them
trainable_embeddings = False
dense = 100
kernels = 100
kernel_length = 3
numbers_of_kappa_measurements = 20
epochs_between_kappa = 10
output = "linear"  # linear, sigmoid or softmax
model_number = 4
essayset = [8]
dropout = 0.5
learning_rates = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
essayfile = "/home/william/m18_edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
wordvectorfile = "/home/william/m18_edvin/Projects/Data/glove.6B/glove.6B.100d.txt"


learning_rates = [0, 0.0001]
numbers_of_kappa_measurements = 2
epochs_between_kappa = 1
number_of_word_embeddings = 1
essayfile = "C:/Users/Edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
wordvectorfile = "C:/Users/Edvin/Projects/Data/glove.6B/glove.6B.100d.txt"


embeddings_index = functions.read_word_vectors(
    wordvectorfile, number_of_word_embeddings
)


data = reader_full.read_dataset(essayset, filepath=essayfile)
data = data[: int(len(data) * 0.7)]  # save 30% of essays for final evaluation
texts, essaysetlist, essaynumber, targets = functions.process_texts(
    data, output, essayset
)
sequences, word_index = functions.texts_to_sequences(MAX_NUM_WORDS, texts)
MAX_SEQUENCE_LENGTH = min(MAX_SEQUENCE_LENGTH, functions.longest_text(sequences))
pad_seq = pad_sequences(
    sequences, maxlen=MAX_SEQUENCE_LENGTH
)  # adds zeros to beginning of text if it is shorter than MAX_SEQUENCE_LENGTH
embedding_layer = functions.embedding_layer(
    MAX_NUM_WORDS,
    MAX_SEQUENCE_LENGTH,
    word_index,
    EMBEDDING_DIM,
    embeddings_index,
    randomize_unseen_words=True,
    trainable=trainable_embeddings,
)
print("Shape of data tensor:", pad_seq.shape)
print("Shape of target tensor:", targets.shape)


for learning_rate in learning_rates:
    print("learning rate ", learning_rate)
    dropoutfolder = (
        "Results/tuning_lr/linear-model4/essayset"
        + str(essayset[0])
        + "/lr"
        + str(learning_rate)
        + "/"
    )
    os.makedirs(dropoutfolder)
    graph_folder = dropoutfolder + "kappa-epoch-graphs/"
    matrix_folder = dropoutfolder + "confusion_matrices/"
    training_folder = dropoutfolder + "/training-values/"
    os.makedirs(graph_folder)
    os.makedirs(matrix_folder)
    os.makedirs(training_folder)

    if output == "softmax":
        model = functions.create_model(
            MAX_SEQUENCE_LENGTH,
            embedding_layer,
            layers=2,
            kernels=kernels,
            kernel_length=kernel_length,
            dense=dense,
            dropout=dropout,
        )
    else:
        model = models.create_model(
            output,
            model_number,
            MAX_SEQUENCE_LENGTH,
            embedding_layer,
            layers=2,
            kernels=kernels,
            kernel_length=kernel_length,
            dense=dense,
            dropout=dropout,
            learning_rate=learning_rate,
        )
    x_train, d_train, x_val, d_val = functions.split_data(
        pad_seq, essaysetlist, essaynumber, targets, VALIDATION_SPLIT
    )
    min_train_loss = 1000
    max_train_acc = 0
    min_val_loss = 1000
    max_val_acc = 0
    max_train_kappa = -1
    max_val_kappa = -1
    epoch = 0
    path = (
        dropoutfolder
        + "/training-values/Dense"
        + str(dense)
        + "Kernels"
        + str(kernels)
        + "Length"
        + str(kernel_length)
    )
    training_file = path + "-training_file.txt"
    training_values = open(training_file, "w+")
    training_values.write(
        "epoch \t train loss \t train acc \t val loss \t val acc \t train kappa \t val kappa \r"
    )

    epoch_list = []
    train_kappa_list = []
    val_kappa_list = []
    for i in range(1, numbers_of_kappa_measurements + 1):
        print("Epoch: " + str(i * epochs_between_kappa))
        model.fit(
            x_train,
            d_train,
            batch_size=10,
            epochs=epochs_between_kappa,
            verbose=False,
            validation_data=(x_val, d_val),
        )
        train_loss, train_acc = model.evaluate(x_train, d_train, verbose=2)
        val_loss, val_acc = model.evaluate(x_val, d_val, verbose=2)
        train_kappa = functions.quadratic_weighted_kappa_for_cnn(
            x_train, d_train, essayset, model, output
        )
        val_kappa = functions.quadratic_weighted_kappa_for_cnn(
            x_val, d_val, essayset, model, output
        )
        training_values.write(
            "%.0f \t %.2f \t  %.2f \t %.2f  \t  %.2f  \t  %.3f  \t  %.3f \r"
            % (
                i * epochs_between_kappa,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                train_kappa,
                val_kappa,
            )
        )

        epoch_list.append(i * epochs_between_kappa)
        train_kappa_list.append(train_kappa)
        val_kappa_list.append(val_kappa)

        if min_train_loss > train_loss:
            min_train_loss = train_loss
        if max_train_acc < train_acc:
            max_train_acc = train_acc
        if min_val_loss > val_loss:
            min_val_loss = val_loss
        if max_val_acc < val_acc:
            max_val_acc = val_acc
        if max_train_kappa < train_kappa:
            max_train_kappa = train_kappa
        if max_val_kappa < val_kappa:
            max_val_kappa = val_kappa
            epoch = i * epochs_between_kappa

    matrix_savefile = (
        matrix_folder
        + "dense"
        + str(dense)
        + "kernels"
        + str(kernels)
        + "kernellength"
        + str(kernel_length)
        + "epochs"
        + str(epochs_between_kappa * numbers_of_kappa_measurements)
    )
    functions.save_confusion_matrix(
        matrix_savefile, x_val, d_val, model, essayset, output
    )
    training_values.close()
    plotimage = (
        graph_folder
        + "Dense"
        + str(dense)
        + "Kernels"
        + str(kernels)
        + "Length"
        + str(kernel_length)
        + ".png"
    )
    functions.plot_kappa(
        plotimage,
        epoch_list,
        train_kappa_list,
        val_kappa_list,
        title="Dropout: " + str(dropout),
        x_axis="Epoch",
    )


print("Done")
