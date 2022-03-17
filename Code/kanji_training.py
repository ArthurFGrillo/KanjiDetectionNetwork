import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import callbacks
from deeplearning_models import kanji_model, kuzushiji_model, keras_model
from my_utils import create_generators_kanji
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == "__main__":

    path_to_train = "E:\\KanjiDataset\\NetworkTraininig\\Train"
    path_to_val = "E:\\KanjiDataset\\NetworkTraininig\\Val"
    path_to_test = "E:\\KanjiDataset\\NetworkTraininig\\Test"

    batch_size = 128
    epochs = 15
    lr = 0.0005

    train_generator, val_generator, test_generator = create_generators_kanji(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    TRAIN = False
    TEST = True

    if TRAIN:
        path_to_save_model = './Models/keras'
        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor="val_accuracy",
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor="val_accuracy",
            patience=5
        )

        #model = kanji_model(nbr_classes)
        #model = kuzushiji_model(nbr_classes)
        model = keras_model(nbr_classes)
        
        optimazer = tf.keras.optimizers.Adam(learning_rate=lr)
        #model.compile(optimizer=optimazer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        model.fit(
            train_generator,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_generator,
            callbacks=[ckpt_saver, early_stop]
        )

    elif TEST:
        modelK = tf.keras.models.load_model('./Models/keras')
        modelJ = tf.keras.models.load_model('./Models/kmnist')
        modelO = tf.keras.models.load_model('./Models/original') 
        
        print("Modelo Keras: ")
        modelK.summary()
        
        print("Evaluating validation set: ")
        modelK.evaluate(val_generator)

        print("Evaluating test set: ")
        modelK.evaluate(test_generator)

        print("Modelo Keras: ")
        modelK.summary()
        
        print("Evaluating validation set: ")
        modelK.evaluate(val_generator)

        print("Evaluating test set: ")
        modelK.evaluate(test_generator)

        print("Modelo Kuzushiji: ")
        modelJ.summary()

        print("Evaluating validation set: ")
        modelJ.evaluate(val_generator)

        print("Evaluating test set: ")
        modelJ.evaluate(test_generator)

        print("Modelo Original: ")
        modelO.summary()

        print("Evaluating validation set: ")
        modelO.evaluate(val_generator)

        print("Evaluating test set: ")
        modelO.evaluate(test_generator)

        