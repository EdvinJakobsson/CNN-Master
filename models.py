from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, Flatten, concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, BatchNormalization, Activation
from keras.models import Model, Sequential
from keras.initializers import Constant
from keras import optimizers, regularizers

def create_model(output, model_number, MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dense = 100, dropout = 0, learning_rate = 0.001, L_two = 0.01, number_of_classes = None):

    if output == 'linear':
        if model_number == 1:
            model = CNN_linear_output1(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout)
        elif model_number == 3:
            model = CNN_linear_output3(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout)
        elif model_number == 4:
            model = CNN_linear_output4(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout, learning_rate = learning_rate)
        elif model_number == 6:
            model = CNN_linear_output6(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout)
        elif model_number == 7:
            model = CNN_linear_output7(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout)
        elif model_number == 8:
            model = CNN_linear_output8(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dense = dense, dropout = dropout, learning_rate = learning_rate, L_two = L_two)
        else:
            print("Create_model function: No linear model was found.")

    elif output == 'sigmoid':
        if model_number == 2:
            model = CNN_sigmoidal_output2(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dropout = dropout)
        elif model_number == 3:
            model = CNN_sigmoidal_output3(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dropout = dropout)
        elif model_number == 4:
            model = CNN_sigmoidal_output4(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dropout = dropout)
        elif model_number == 6:
            model = CNN_sigmoidal_output6(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dropout = dropout)
        elif model_number == 7:
            model = CNN_sigmoidal_output7(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dropout = dropout)
        else:
            print("Create_model function: No sigmoid model was found.")

    elif output == 'softmax':
        if model_number == 4:
            model = CNN_softmax_output4(number_of_classes, MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dropout = dropout, learning_rate = learning_rate)
        elif model_number == 8:
            model = CNN_softmax_output8(number_of_classes, MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = kernels, kernel_length = kernel_length, dropout = dropout, learning_rate = learning_rate, L_two = L_two)
        else:
            print("Create_model function: No model was found.")
    elif output == 'hybrid':
        if model_number == 1:
            model = hybrid1(MAX_SEQUENCE_LENGTH, embedding_layer)
        if model_number == 2:
            model = hybrid2(MAX_SEQUENCE_LENGTH, embedding_layer)
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
def CNN_linear_output4(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dense = 128, dropout = 0, learning_rate = 0.001):
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
    opt = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    print("model 4 linear created")
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


    #added L2 to model 4
def CNN_linear_output8(MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dense = 128, dropout = 0, learning_rate = 0.001, L_two = 0.01):
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
    model.add(Dense(dense, activation='sigmoid', kernel_regularizer=regularizers.l2(L_two)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    opt = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    model.summary()
    exit(0)
    return model






######################       SOFTMAX      ################################


    #softmax without globalmaxpooling
def CNN_softmax_output4(number_of_classes, MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dense = 128, dropout = 0, learning_rate = 0.001):
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
    model.add(Dense(number_of_classes, activation='softmax'))
    opt = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print("model 4 softmax created")
    return model



    #added L2 to model 4
def CNN_softmax_output8(number_of_classes, MAX_SEQUENCE_LENGTH, embedding_layer, layers = 2, kernels = 1, kernel_length = 1, dense = 128, dropout = 0, learning_rate = 0.001, L_two = 0.01):
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
    model.add(Dense(dense, activation='sigmoid', kernel_regularizer=regularizers.l2(L_two)))
    model.add(Dropout(dropout))
    model.add(Dense(number_of_classes, activation='softmax'))
    opt = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


    return model




##########################       HYBRID      ##############################

def hybrid1(MAX_SEQUENCE_LENGTH, embedding_layer):

    cnn_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='cnn_input')
    embedded_sequences = embedding_layer(cnn_input)
    x1 = Conv1D(100, 3, activation='relu')(embedded_sequences)
    x1 = MaxPooling1D(5)(x1)
    x1 = Conv1D(100, 3, activation='relu')(x1)
    x1 = MaxPooling1D(5)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(100, activation='sigmoid')(x1)

    mlp_input = Input(shape=(4,), name='mlp_input')
    x2 = Dense(50, activation='relu')(mlp_input)
    x2 = Dense(50, activation='relu')(x2)
    x2 = Dense(50, activation='relu')(x2)

    x = concatenate([x1, x2])
    x = Dense(100, activation='relu')(x)

    prediction = Dense(1, activation='linear', name='prediction')(x)
    model = Model(inputs=[cnn_input, mlp_input], outputs=prediction)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    return model
