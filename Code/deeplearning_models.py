import tensorflow
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, MaxPooling2D, Dropout
from tensorflow.keras import Model
from keras.models import Sequential

def kanji_model(nbr_classes):

    #Lembrar de Mudar isso Quando eu For Usar !!!
    my_input = Input(shape=(36, 36, 1)) #ajustar o tamanha das fotos do modelo

    x = Conv2D(128, (3,3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(nbr_classes, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

def kuzushiji_model(nbr_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(36, 36, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nbr_classes, activation='softmax'))
    return model

def keras_model(nbr_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(36, 36, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(nbr_classes, activation="softmax"))
    return model

def modelo_original(numero_classes):

    model = Sequential()

    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(36, 36, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(numero_classes, activation='softmax'))
    
    return model

if __name__=='__main__':
    model = kanji_model(10)
    model.summary()
