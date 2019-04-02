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


embeddings_index = functions.read_word_vectors()

data = reader_full.read_dataset(0,1246)

texts, essayset, essaynumber, targets = functions.process_texts(data)

sequences, word_index = functions.texts_to_sequences(MAX_NUM_WORDS, texts)

MAX_SEQUENCE_LENGTH = min(MAX_SEQUENCE_LENGTH, functions.longest_text(sequences))

pad_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) #adds zeros to beginning of text if it is shorter than MAX_SEQUENCE_LENGTH

print('Shape of data tensor:', pad_sequences.shape)
print('Shape of target tensor:', targets.shape)

dense_numbers = [1,2,3,5,7,10,20,50]
for dense in dense_numbers:
    print("Dense: ", dense)
    file = "Results/layers2dense" + str(dense) + ".txt"
    f = open(file, "w+")
    f.write("essays: 1246 \t \t epochs: 200 \t \t  Dropout: no \t \t k-fold: no \t \t batch size: 128 \t\t layers: 2 \t \t dense: " + str(dense) + " \r \r")

    for kernel_length in range(1,6):
        print("kernel length: ", kernel_length)
        f.write("Kernel length  \t kernels \t min train loss \t top train acc \t min val loss \t top val acc \t top train kappa \t top val kappa \t epoch at top val kappa \r")

        # split the data into a training set and a validation set
        kernel_numbers = [10,50,100]
        for kernels in kernel_numbers:
            print("kernels: ", kernels)
            x_train, d_train, x_val, d_val = functions.split_data(pad_sequences, essayset, essaynumber, targets, VALIDATION_SPLIT)
            embedding_layer = functions.embedding_layer(MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH, word_index, EMBEDDING_DIM, embeddings_index, randomize_unseen_words = True, trainable = False)
            model = functions.create_model( MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense)

            min_train_loss = 1000
            max_train_acc = 0
            min_val_loss = 1000
            max_val_acc = 0
            max_train_kappa = -1
            max_val_kappa = -1
            epoch = 0

            path = "Results/Images/dense" + str(dense) + "kernels" + str(kernels) + "kernellength" + str(kernel_length)
            os.makedirs(path)

            for epochs_between_kappa in range(3):

                model.fit(x_train, d_train, batch_size=10, epochs=20, verbose=False, validation_data=(x_val, d_val))
                train_loss, train_acc = model.evaluate(x_train, d_train, verbose=2)
                val_loss, val_acc = model.evaluate(x_val, d_val, verbose=2)
                train_kappa = functions.quadratic_weighted_kappa_for_cnn(x_train, d_train, model)
                val_kappa = functions.quadratic_weighted_kappa_for_cnn(x_val, d_val, model)

                savefile = path + "/dense" + str(dense) + "kernels" + str(kernels) + "kernellength" + str(kernel_length) + "epochs" + str((epochs_between_kappa+1)*10)
                functions.save_confusion_matrix(savefile, model, x_val, d_val, lowest_score=2, highest_score=12, title=None)

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
                    epoch = 10*(epochs_between_kappa+1)


            f.write("%.0f \t %.0f \t %.2f \t  %.2f \t %.2f  \t  %.2f  \t  %.3f  \t  %.3f  \t  %.0f \r" % (kernel_length, kernels, min_train_loss, max_train_acc, min_val_loss, max_val_acc, max_train_kappa, max_val_kappa, epoch))
    f.close()

print("Done")
