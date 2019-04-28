from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, BatchNormalization, Activation
from keras.models import Model, Sequential
from keras.initializers import Constant


def CNN_sigmoidal_output(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dropout = 0):

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = Dropout(dropout)(embedded_sequences)
    x = Conv1D(kernels, kernel_length, activation='relu')(embedded_sequences)
    for layers in range(1, layers):
        x = MaxPooling1D(5)(x)
        x = Conv1D(kernels, kernel_length, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    y = Dense(1, activation='sigmoid')(x)

    model = Model(sequence_input, y)
    model.compile(loss='mse',
                  optimizer='rmsprop', metrics=['acc'])
    return model


        #no batch normalization, dropout on embedding, GlobalMaxPooling
def CNN_sigmoidal_output2(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dropout = 0):

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(dropout))
    model.add(Conv1D(kernels, kernel_length, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(kernels, kernel_length, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer="rmsprop", metrics=['accuracy'])

    return model




        #batch normalization and global maxpooling
def CNN_sigmoidal_output3(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dropout = 0):

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(dropout))
    model.add(Conv1D(kernels, kernel_length))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(kernels, kernel_length))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer="rmsprop", metrics=['accuracy'])

    return model



        #dropout at end with dense layer instead
def CNN_sigmoidal_output4(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dropout = 0):

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(kernels, kernel_length))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(kernels, kernel_length))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(5))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    return model


        #dropout at end with dense layer instead, two dense layers
    def CNN_sigmoidal_output5(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dropout = 0):

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        model = Sequential()
        model.add(embedding_layer)
        model.add(Conv1D(kernels, kernel_length))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(kernels, kernel_length))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling1D(5))
        model.add(Flatten())
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dropout(dropout))
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

        return model
