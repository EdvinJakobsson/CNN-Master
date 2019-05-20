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
from BenHamner.score import mean_quadratic_weighted_kappa
import time
start = time.time()

#dont change these
asap_ranges = {
0: (0, 60),
1: (2, 12),
2: (1, 6),
3: (0, 3),
4: (0, 3),
5: (0, 4),
6: (0, 4),
7: (0, 30),
8: (0, 60)}
human_range = False     #set to true if comparison should be made with one expert human grader instead of the resolved score of the two experts
if human_range == True:
    print("WARNING: human_range is set to True!")
    print("WARNING: human_range is set to True!")
    print("WARNING: human_range is set to True!")
    asap_ranges = {    0: (0, 60),    1: (1, 6),    2: (1, 6),    3: (0, 3),    4: (0, 3),    5: (0, 4),    6: (0, 4),    7: (0, 15),    8: (0, 30)}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    #removes some of the tf warnings
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 100000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
randomize_data = False
essayfile = "/home/william/m18_edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
wordvectorfile = "/home/william/m18_edvin/Projects/Data/glove.6B/glove.6B.100d.txt"

#word embeddings
number_of_word_embeddings = -1       #-1 = all of them
trainable_embeddings = True

#hyper-parameters
output = 'linear'      #linear, sigmoid or softmax
model_number = 8
dropout = 0.5
learning_rate = 0.0001
dense = 100
kernels = 100
kernel_length = 3
batch_size = 10
tests = 3

#training
epochs_between_kappa = 1
essaysets = [[1],[2],[3],[4],[5],[6],[7],[8]]
L_two = 0.0001


# #test values only for my computer
# number_of_word_embeddings = 1
# dense = 1
# kernels = 1
# kernel_length = 1
# essaysets = [[1]]
# essayfile = "C:/Users/Edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
# wordvectorfile = "C:/Users/Edvin/Projects/Data/glove.6B/glove.6B.100d.txt"
# tests = 2


embeddings_index = functions.read_word_vectors(wordvectorfile,number_of_word_embeddings)

basefolder = "Results/" + output + str(model_number) + "/"
os.makedirs(basefolder)
kappa_file = open(basefolder + "kappas.txt", "+w")

for test in range(1,tests+1):
    print("test ", test)
    folder = basefolder + "test" + str(test) + "/"
    os.makedirs(folder)
    kappa_list = []
    for essayset in essaysets:
        print("essayset ", essayset[0])
        set = essayset[0]
        number_of_kappa_measurements = 50
        batch_size = 40
        if set == 7:
            number_of_kappa_measurements = 200
        if set == 8:
            number_of_kappa_measurements = 200
            batch_size = 10

        number_of_classes =  asap_ranges[set][1]-asap_ranges[set][0]+1
        data = reader_full.read_dataset(essayset, filepath=essayfile)
        data = data[:int(len(data)*0.7)]    # save 30% of essays for final evaluation
        texts, essaysetlist, essaynumber, targets = functions.process_texts(data, output, essayset, asap_ranges, human_range)
        sequences, word_index = functions.texts_to_sequences(MAX_NUM_WORDS, texts)
        MAX_SEQUENCE_LENGTH = min(MAX_SEQUENCE_LENGTH, functions.longest_text(sequences))
        pad_seq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) #adds zeros to beginning of text if it is shorter than MAX_SEQUENCE_LENGTH
        embedding_layer = functions.embedding_layer(MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH, word_index, EMBEDDING_DIM, embeddings_index, randomize_unseen_words = True, trainable = trainable_embeddings)

        model = models.create_model(output, model_number, MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout, learning_rate = learning_rate, L_two = L_two, number_of_classes = number_of_classes)
        x_train, d_train, x_val, d_val = functions.split_data(pad_seq, essaysetlist, essaynumber, targets, VALIDATION_SPLIT, randomize_data)

        training_file = folder + "set" + str(set) + "-training_file.txt"
        training_values = open(training_file, "w+")
        training_values.write("epoch \t train loss \t train acc \t val loss \t val acc \t train kappa \t val kappa \r")

        epoch_list = []
        train_loss_list = []
        val_loss_list = []
        train_kappa_list = []
        val_kappa_list = []
        for i in range(1, number_of_kappa_measurements+1):
            #print("Epoch: " + str(i*epochs_between_kappa))
            model.fit(x_train, d_train, batch_size=batch_size, epochs=epochs_between_kappa, verbose=False, validation_data=(x_val, d_val))
            train_loss, train_acc = model.evaluate(x_train, d_train, verbose=2)
            val_loss, val_acc = model.evaluate(x_val, d_val, verbose=2)
            train_kappa = functions.quadratic_weighted_kappa_for_cnn(x_train, d_train, essayset, model, output)
            val_kappa = functions.quadratic_weighted_kappa_for_cnn(x_val, d_val, essayset, model, output)

            training_values.write("%.0f \t %.2f \t  %.2f \t %.2f  \t  %.2f  \t  %.3f  \t  %.3f \r" % (i*epochs_between_kappa, train_loss, train_acc, val_loss, val_acc, train_kappa, val_kappa))
            epoch_list.append(i*epochs_between_kappa)
            train_kappa_list.append(train_kappa)
            val_kappa_list.append(val_kappa)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

        training_values.close()
        matrix_savefile = folder + "Confusion-Matrix-set" + str(set) + ".png"
        functions.save_confusion_matrix(matrix_savefile, x_val, d_val, model, essayset, output, asap_ranges, "Confusion Matrix")
        kappa_plot = folder + "kappa-set" + str(set) + ".png"
        functions.plot_kappa(kappa_plot, epoch_list, val_kappa_list, train_kappa_list, title = "Dropout: " + str(dropout), x_axis="Epoch")

        loss_ymax = 2
        if essayset[0] == 7:
            loss_ymax = 20
        if essayset[0] == 8:
            loss_ymax = 40
        if output == 'softmax':
            loss_ymax = 2
        loss_plot = folder + "loss-set" + str(set) + ".png"
        functions.plot_loss(loss_plot, epoch_list, train_loss_list, val_loss_list, title = "Dropout: " + str(dropout), x_axis="Epoch", y_max = loss_ymax)


        kappa_list.append(val_kappa_list[-1])
        kappa_file.write(str(val_kappa_list[-1]) + "\t")
    essay_set_list = [i+1 for i in range(len(kappa_list))]
    functions.plot_kappa(folder + "kappas.png", essay_set_list, kappa_list, title = "Dropout: " + str(dropout), x_axis="Essay Set")
    mean_kappa = mean_quadratic_weighted_kappa(kappa_list)
    kappa_file.write("\t \t \t" + str(mean_kappa) + "\r")



kappa_file.close()
end = time.time()

if human_range == True:
    print("WARNING: human_range is set to True!")
    print("WARNING: human_range is set to True!")
    print("WARNING: human_range is set to True!")

time = end-start
hours = int(time/3600)
min = int((time-3600*hours)/60)
sec = int(time - hours*3600 - min*60)
print("Run-Time: " + str(hours) + " hours, " + str(min) + " minutes, " + str(sec) + " seconds.")
print("Done")
