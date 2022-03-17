import tensorflow as tf
import numpy as np
import csv
import glob
import os
import subprocess
import time
from PIL import ImageGrab, Image, ImageOps
from my_utils import ajust_images, clean_trial_folder
from tensorflow.python.ops.gen_math_ops import imag

def predict_with_model(model, img_path): # Predic using the saved model
    
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [36,36])
    image = tf.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predictions = np.argmax(predictions)

    return predictions.item()
    #return predictions

def predict_with_model_speed(model, img_path): # Predic using the saved model
    
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [36,36])
    image = tf.expand_dims(image, axis=0)

    start = time.time()
    predictions = model.predict(image)
    end = time.time()

    result = end - start
    return result

def create_dictionary(file_name): # create a dictionary using the given csv file
    with open(file_name, encoding="utf8") as file:
        reader = csv.reader(file)
        myDict = {}
        for row in reader:
            key = int(row.pop(0))
            myDict[key] = row
        file.close
        return myDict

def print_info(myDict, answer): # answer with a user friendly message
    info_list = myDict[answer]
    print("\nDetected Kanji:", info_list[0], " Meaning:", info_list[3], "Siginificado:", info_list[4])
    print("It's pronunciation in Kana is:", info_list[5])
    print("It's pronunciation in Romanji is:", info_list[6])

def predict_folder(path_to_raw, path_to_done, model, dictionary): # Ajust n predict all images in a folder 
    ajust_images(path_to_raw, path_to_done) # posso acelerar esse precesso fazendo o ajuste e a predicao no mesmo for !!!

    for file_name in glob.glob(os.path.join(path_to_done,'*.png')):
        val = predict_with_model(model,file_name)
        print_info(dictionary, val)

    clean_trial_folder(path_to_done)

def Test_speed(path_to_folder, model, dictionary):
    sum = 0
    num = 0
    for file_name in glob.glob(os.path.join(path_to_folder,'*.png')):
        num += 1

        val = predict_with_model_speed(model,file_name)
        sum = sum + val
        
        text = ["Guess: ", val]
        print(text)   

    text = ["Average: ", sum/num]
    print(text)

def create_image_from_clipboard(save_path):
    try:
        clip_img = ImageGrab.grabclipboard()
        list = os.listdir(save_path)
        num_files = len(list)
        img_name = ((num_files//2)+1)
        path_raw = save_path + "/Raw{}.png".format(img_name)
        clip_img.save(path_raw,'PNG') 

        edited_img = ImageOps.grayscale(clip_img)
        edited_img = edited_img.resize((36,36), Image.ANTIALIAS)
        path_net = save_path + "/Net{}.png".format(img_name)
        edited_img.save(path_net,'PNG')

        return [path_raw, path_net] 
    except AttributeError as error:
        print("\nThe Content on The ClipBoard is Not a Image")
        return None
    except Exception as exception:
        print("\nA Unexpected Exception Has Appeared")
        return None

if __name__ == "__main__":

    # Geting the folder paths
    image_folder = './Temp/Images'

    # loading the model
    modelK = tf.keras.models.load_model('./Models/keras')
    modelJ = tf.keras.models.load_model('./Models/kmnist')
    modelO = tf.keras.models.load_model('./Models/original') 

    # Answering the dictionary
    name  = "./Resources/KanjiTable.csv"
    kanjiDict = create_dictionary(name)
    file_test = './Resources/default_done.png'


    print("\nResultados Modelo Original")
    predict_with_model(modelO, file_test)
    Test_speed(image_folder, modelO, kanjiDict)

    print("\nResultados Modelo Kuzushiji")
    predict_with_model(modelJ, file_test)
    Test_speed(image_folder, modelJ, kanjiDict)

    print("\nResultados Modelo Keras")
    predict_with_model(modelK, file_test)
    Test_speed(image_folder, modelK, kanjiDict)

    # start = time.time()
    # time.sleep(1)
    # end = time.time()
    # print(end - start)

    # finish = False
    # while(not finish):
    #     print("\nPress 'Windows + Shift + S' or type 'A' to select a Kanji and then Press ENTER to add it to the folder")
    #     print("Once you are done Press 'F' to Finish the input and run the detection")

    #     #Finish if F
    #     answer = input("\nWaiting For Input: ")
    #     if answer == "F" or answer == "f":
    #         finish = True
    #         break

    #     if answer == "A" or answer == "a":
    #         subprocess.run(["explorer.exe", "ms-screenclip:"])
    #         continue

    #     # Adding CLipBoard to folder
    #     result = create_image_from_clipboard(image_folder)
    #     if(result != None):
    #         print("Image Added Successfully")
    #         print(result[0])
    #         print(result[1])
    #         val = predict_with_model(modelo,result[1])
    #         print_info(kanjiDict, val)       
    #     else:
    #         print("Image NOT Added Successfully!!!")

    # clean_trial_folder(image_folder)

    print("\nEnd...")
