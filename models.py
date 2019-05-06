from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, BatchNormalization, Activation
from keras.models import Model, Sequential
from keras.initializers import Constant


def create_model(output, model_number, MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dense = 100, dropout = 0):

    if output == 'linear':
        if model_number == 1:
            model = CNN_linear_output1(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout)
        if model_number == 3:
            model = CNN_linear_output3(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout)
        if model_number == 4:
            model = CNN_linear_output4(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout)
        if model_number == 6:
            model = CNN_linear_output6(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout)
        if model_number == 7:
            model = CNN_linear_output7(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout)

    elif output == 'sigmoid':
        if model_number == 2:
            model = CNN_sigmoidal_output2(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dropout = dropout)
        if model_number == 3:
            model = CNN_sigmoidal_output3(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dropout = dropout)
        if model_number == 4:
            model = CNN_sigmoidal_output4(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dropout = dropout)
        if model_number == 6:
            model = CNN_sigmoidal_output6(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dropout = dropout)
        if model_number == 7:
            model = CNN_sigmoidal_output7(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dropout = dropout)

    else:
        print("Create_model function: No model was found.")
    return model


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
    print("model 2 sigmoid complete")
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
    print("model 3 sigmoid complete")
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
    print("model 4 sigmoid complete")
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


    #dropout at end with dense layer with few nodes
def CNN_sigmoidal_output6(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dropout = 0):

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
    model.add(Dense(12, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    return model

    #no batch normalizaton, but 128 dense layer
def CNN_sigmoidal_output7(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dropout = 0):

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(kernels, kernel_length))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(kernels, kernel_length))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(5))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    return model




#####################     LINEAR    ######################


    #adjustable dense layer at end. Batch Normalization
def CNN_linear_output1(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dense = 100, dropout = 0):

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
    model.add(Dense(dense, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    return model

    #batch normalization and global maxpooling
def CNN_linear_output3(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dense = 1, dropout = 0):

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
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer="rmsprop", metrics=['accuracy'])

    return model

    #same as model 1
def CNN_linear_output4(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dense = 128, dropout = 0):

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
    model.add(Dense(dense, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    print("model 4 linear complete")
    return model

    #dropout at end with dense layer with few nodes
def CNN_linear_output6(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dense = 12, dropout = 0):

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
    model.add(Dense(12, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    return model

    #no batch normalization
def CNN_linear_output7(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dense = 100, dropout = 0):

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(kernels, kernel_length))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(kernels, kernel_length))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(5))
    model.add(Flatten())
    model.add(Dense(dense, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    return model
