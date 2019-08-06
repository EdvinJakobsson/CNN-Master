import os
import sys
import numpy as np
import csv
import time

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from cnn_master.ben_hamner.score import mean_quadratic_weighted_kappa

from cnn_master import reader_full, functions, models
from cnn_master.config import Config
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # removes some of the tf warnings


def setup_folder():
    basefolder = "Results/" + Config.MODEL_OUTPUT + str(Config.MODEL_NUMBER) + "/"
    os.makedirs(basefolder)
    kappa_file = open(basefolder + "kappas.txt", "+w")
    return basefolder, kappa_file

def main():
    #start = time.time()
    embeddings_index = functions.read_word_vectors(
        Config.WORDVECTOR_FILE, Config.NUM_WORD_EMBEDDINGS
    )
    basefolder, kappa_file = setup_folder()

    for iteration in range(1, Config.ITERATIONS + 1):
        print("iteration ", iteration)
        folder = basefolder + "iteration" + str(iteration) + "/"
        os.makedirs(folder)
        kappa_list = []
        for essayset in Config.ESSAYSETS:
            print("essayset ", essayset[0])
            set = essayset[0]
            number_of_kappa_measurements = 50
            batch_size = 40
            if set == 7:
                number_of_kappa_measurements = 200
            if set == 8:
                number_of_kappa_measurements = 200
                batch_size = 10

            number_of_classes = Config.ASAP_RANGES[set][1] - Config.ASAP_RANGES[set][0] + 1
            data = reader_full.read_dataset(essayset, filepath=Config.ESSAY_FILE)
            data = data[: int(len(data) * 0.7)]  # save 30% of essays for final evaluation
            texts, essaysetlist, essaynumber, targets = functions.process_texts(
                data, Config.MODEL_OUTPUT, essayset, Config.ASAP_RANGES, Config.HUMAN_RANGE
            )
            sequences, word_index = functions.texts_to_sequences(Config.MAX_NUM_WORDS, texts)
            Config.MAX_SEQUENCE_LENGTH = min(
                Config.MAX_SEQUENCE_LENGTH, functions.longest_text(sequences)
            )
            pad_seq = pad_sequences(
                sequences, maxlen=Config.MAX_SEQUENCE_LENGTH
            )  # adds zeros to beginning of text if it is shorter than Config.MAX_SEQUENCE_LENGTH
            embedding_layer = functions.embedding_layer(
                Config.MAX_NUM_WORDS,
                Config.MAX_SEQUENCE_LENGTH,
                word_index,
                Config.EMBEDDING_DIM,
                embeddings_index,
                randomize_unseen_words=True,
                trainable=Config.TRAINABLE_EMBEDDINGS,
            )

            model = models.create_model(
                Config.MODEL_OUTPUT,
                Config.MODEL_NUMBER,
                Config.MAX_SEQUENCE_LENGTH,
                embedding_layer,
                layers=2,
                kernels=Config.KERNELS,
                kernel_length=Config.KERNEL_LENGTH,
                dense=Config.DENSE,
                dropout=Config.DROPOUT,
                learning_rate=Config.LEARNING_RATE,
                L_two=Config.L_TWO,
                number_of_classes=number_of_classes,
            )
            x_train, d_train, x_val, d_val = functions.split_data(
                pad_seq,
                essaysetlist,
                essaynumber,
                targets,
                Config.VALIDATION_SPLIT,
                Config.RANDOMIZE_DATA,
            )

            training_file = folder + "set" + str(set) + "-training_file.txt"
            training_values = open(training_file, "w+")
            training_values.write(
                "epoch \t train loss \t train acc \t val loss \t val acc \t train kappa \t val kappa \r"
            )

            epoch_list = []
            train_loss_list = []
            val_loss_list = []
            train_kappa_list = []
            val_kappa_list = []

            for i in range(1, number_of_kappa_measurements + 1):
                model.fit(
                    x_train,
                    d_train,
                    batch_size=batch_size,
                    epochs=Config.EPOCHS_BETWEEN_KAPPA,
                    verbose=False,
                    validation_data=(x_val, d_val),
                )
                train_loss, train_acc = model.evaluate(x_train, d_train, verbose=2)
                val_loss, val_acc = model.evaluate(x_val, d_val, verbose=2)
                train_kappa = functions.quadratic_weighted_kappa_for_cnn(
                    x_train, d_train, essayset, model, Config.MODEL_OUTPUT
                )
                val_kappa = functions.quadratic_weighted_kappa_for_cnn(
                    x_val, d_val, essayset, model, Config.MODEL_OUTPUT
                )

                training_values.write(
                    "%.0f \t %.2f \t  %.2f \t %.2f  \t  %.2f  \t  %.3f  \t  %.3f \r"
                    % (
                        i * Config.EPOCHS_BETWEEN_KAPPA,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                        train_kappa,
                        val_kappa,
                    )
                )
                epoch_list.append(i * Config.EPOCHS_BETWEEN_KAPPA)
                train_kappa_list.append(train_kappa)
                val_kappa_list.append(val_kappa)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)

            training_values.close()
            matrix_savefile = folder + "Confusion-Matrix-set" + str(set) + ".png"
            functions.save_confusion_matrix(
                matrix_savefile,
                x_val,
                d_val,
                model,
                essayset,
                Config.MODEL_OUTPUT,
                Config.ASAP_RANGES,
                "Confusion Matrix",
            )
            kappa_plot = folder + "kappa-set" + str(set) + ".png"
            functions.plot_kappa(
                kappa_plot,
                epoch_list,
                val_kappa_list,
                train_kappa_list,
                title="Dropout: " + str(Config.DROPOUT),
                x_axis="Epoch",
            )

            loss_ymax = 2
            if essayset[0] == 7:
                loss_ymax = 20
            if essayset[0] == 8:
                loss_ymax = 40
            if Config.MODEL_OUTPUT == "softmax":
                loss_ymax = 2
            loss_plot = folder + "loss-set" + str(set) + ".png"
            functions.plot_loss(
                loss_plot,
                epoch_list,
                train_loss_list,
                val_loss_list,
                title="Dropout: " + str(Config.DROPOUT),
                x_axis="Epoch",
                y_max=loss_ymax,
            )

            kappa_list.append(val_kappa_list[-1])
            kappa_file.write(str(val_kappa_list[-1]) + "\t")
        essay_set_list = [i + 1 for i in range(len(kappa_list))]
        functions.plot_kappa(
            folder + "kappas.png",
            essay_set_list,
            kappa_list,
            title="Dropout: " + str(Config.DROPOUT),
            x_axis="Essay Set",
        )
        mean_kappa = mean_quadratic_weighted_kappa(kappa_list)
        kappa_file.write("\t \t \t" + str(mean_kappa) + "\r")


    kappa_file.close()
    # end = time.time()
    #
    # if Config.HUMAN_RANGE == True:
    #     print("WARNING: human_range is set to True!")
    #
    # time = end - start
    # hours = int(time / 3600)
    # min = int((time - 3600 * hours) / 60)
    # sec = int(time - hours * 3600 - min * 60)
    # print(
    #     "Run-Time: "
    #     + str(hours)
    #     + " hours, "
    #     + str(min)
    #     + " minutes, "
    #     + str(sec)
    #     + " seconds."
    # )
    print("Done")

if __name__ == "__main__":
    main()
