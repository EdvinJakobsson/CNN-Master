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


MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 100000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
essayset = [1]
number_of_word_embeddings = -1   #all of them

softmax_output = False
output = 'linear'       #linear, sigmoid or softmax
model_numbers = [1]
trainable_embeddings = False
dense_numbers = [100]
kernel_numbers = [100]
kernel_length_number = [3]
numbers_of_kappa_measurements = 20
epochs_between_kappa = 10
dropout_numbers = [0, 0.5, 0.99]
#dropout_numbers = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
#dropout_numbers = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

essayfile = "/home/william/m18_edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
wordvectorfile = "/home/william/m18_edvin/Projects/Data/glove.6B/glove.6B.100d.txt"



# output = 'linear'       #linear, sigmoid or softmax
# dense_numbers = [1]
# model_numbers = [1]
# essayset = [1]
# kernel_numbers = [1]
# kernel_length_number = [3]
# numbers_of_kappa_measurements = 2
# epochs_between_kappa = 1
# dropout_numbers = [0]
# number_of_word_embeddings = 1
# essayfile = "C:/Users/Edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
# wordvectorfile = "C:/Users/Edvin/Projects/Data/glove.6B/glove.6B.100d.txt"


embeddings_index = functions.read_word_vectors(wordvectorfile,number_of_word_embeddings)
data = reader_full.read_dataset(essayset, filepath=essayfile)
data = data[:int(len(data)*0.7)]    # save 30% of essays for final evaluation
texts, essaysetlist, essaynumber, targets = functions.process_texts(data, output, essayset)
sequences, word_index = functions.texts_to_sequences(MAX_NUM_WORDS, texts)
MAX_SEQUENCE_LENGTH = min(MAX_SEQUENCE_LENGTH, functions.longest_text(sequences))
pad_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) #adds zeros to beginning of text if it is shorter than MAX_SEQUENCE_LENGTH


print('Shape of data tensor:', pad_sequences.shape)
print('Shape of target tensor:', targets.shape)



for model_number in model_numbers:
    model_folder = "Results/model" + str(model_number)
    os.makedirs(model_folder + "/graphs")
    print("model number: ", model_number)
    train_kappa_dropout_list = []
    val_kappa_dropout_list = []

    dropout_values = open(model_folder + "/dropout_values.txt", "w+")
    dropout_values.write("essays: 1248 \t \t epochs: " + str(numbers_of_kappa_measurements*epochs_between_kappa) + " \t \t k-fold: no \t \t batch size: 10 \t\t layers: 2 \t \t softmax_output: " + str(softmax_output) + " \r \r")
    dropout_values.write("Dropout \t Kernel length  \t kernels \t min train loss \t top train acc \t min val loss \t top val acc \t top train kappa \t top val kappa \t epoch at top val kappa \r")

    for dropout in dropout_numbers:
        print("Dropout: ", dropout)
        folder = "Results/model" + str(model_number) + "/dropout" + str(dropout) + "/"
        os.makedirs(folder)

        for dense in dense_numbers:
            print("Dense: ", dense)
            file = folder + "layers2dense" + str(dense) + ".txt"
            f = open(file, "w+")
            f.write("essays: 1248 \t \t epochs: " + str(numbers_of_kappa_measurements*epochs_between_kappa) + " \t \t  Dropout: " + str(dropout) + " \t \t k-fold: no \t \t batch size: 10 \t\t layers: 2 \t \t softmax_output: " + str(softmax_output) + " \t \t dense: " + str(dense) + " \r \r")

            for kernel_length in kernel_length_number:
                print("kernel length: ", kernel_length)
                f.write("Kernel length  \t kernels \t min train loss \t top train acc \t min val loss \t top val acc \t top train kappa \t top val kappa \t epoch at top val kappa \r")

                # split the data into a training set and a validation set
                for kernels in kernel_numbers:
                    print("kernels: ", kernels)
                    x_train, d_train, x_val, d_val = functions.split_data(pad_sequences, essaysetlist, essaynumber, targets, VALIDATION_SPLIT)
                    embedding_layer = functions.embedding_layer(MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH, word_index, EMBEDDING_DIM, embeddings_index, randomize_unseen_words = True, trainable = trainable_embeddings)

                    if softmax_output:
                        model = functions.create_model(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout)
                    else:
                        model = models.create_model(output, model_number, MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout)


                    min_train_loss = 1000
                    max_train_acc = 0
                    min_val_loss = 1000
                    max_val_acc = 0
                    max_train_kappa = -1
                    max_val_kappa = -1
                    epoch = 0

                    path = folder + "Images/dense" + str(dense) + "kernels" + str(kernels) + "kernellength" + str(kernel_length)
                    os.makedirs(path)
                    training_file = path + "/training_file.txt"
                    training_values = open(training_file, "w+")
                    training_values.write("epoch \t train loss \t train acc \t val loss \t val acc \t train kappa \t val kappa \r")

                    epoch_list = []
                    train_kappa_list = []
                    val_kappa_list = []
                    for i in range(1, numbers_of_kappa_measurements+1):
                        print("Epoch: " + str(i*epochs_between_kappa))
                        model.fit(x_train, d_train, batch_size=10, epochs=epochs_between_kappa, verbose=False, validation_data=(x_val, d_val))
                        train_loss, train_acc = model.evaluate(x_train, d_train, verbose=2)
                        val_loss, val_acc = model.evaluate(x_val, d_val, verbose=2)
                        train_kappa = functions.quadratic_weighted_kappa_for_cnn(x_train, d_train, essayset, model, output)
                        val_kappa = functions.quadratic_weighted_kappa_for_cnn(x_val, d_val, essayset, model, output)
                        training_values.write("%.0f \t %.2f \t  %.2f \t %.2f  \t  %.2f  \t  %.3f  \t  %.3f \r" % (i*epochs_between_kappa, train_loss, train_acc, val_loss, val_acc, train_kappa, val_kappa))
                        #savefile = path + "/dense" + str(dense) + "kernels" + str(kernels) + "kernellength" + str(kernel_length) + "epochs" + str(epochs_between_kappa * i)
                        #functions.save_confusion_matrix(savefile, x_val, d_val, model, essayset, softmax_output)

                        epoch_list.append(i*epochs_between_kappa)
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
                            epoch = i*epochs_between_kappa

                    train_kappa_dropout_list.append(max_train_kappa)
                    val_kappa_dropout_list.append(max_val_kappa)
                    f.write("%.0f \t %.0f \t %.2f \t  %.2f \t %.2f  \t  %.2f  \t  %.3f  \t  %.3f  \t  %.0f \r" % (kernel_length, kernels, min_train_loss, max_train_acc, min_val_loss, max_val_acc, max_train_kappa, max_val_kappa, epoch))

                    training_values.close()
                    plotimage = "Results/model" + str(model_number) + "/graphs/" + str(dropout*20) + ".png"
                    functions.plot_kappa(plotimage, epoch_list, train_kappa_list, val_kappa_list, title = "Dropout: " + str(dropout), x_axis="Epoch")
            f.close()
        dropout_values.write("%.3f \t %.0f \t %.0f \t %.2f \t  %.2f \t %.2f  \t  %.2f  \t  %.3f  \t  %.3f  \t  %.0f \r" % (dropout, kernel_length, kernels, min_train_loss, max_train_acc, min_val_loss, max_val_acc, max_train_kappa, max_val_kappa, epoch))
    dropout_values.close()
    functions.plot_kappa("Results/model" + str(model_number) + "/dropout_Kappa_graph.png", dropout_numbers, train_kappa_dropout_list, val_kappa_dropout_list, title= "Overall Kappa", x_axis="Dropout")
print("Done")
