import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps

def create_generators_kanji(batch_size, train_data_path, val_data_path, test_data_path): # Create Kanji Data Generators
    
    train_preprocessor = ImageDataGenerator(
        rescale= 1/255.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8,1.2],
        zoom_range=[0.7,1.3],
    )

    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(36,36),
        color_mode='grayscale',
        shuffle=True,
        batch_size=batch_size
    )

    test_preprocessor = ImageDataGenerator(
        rescale= 1/255.,
        #rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        #brightness_range=[0.8,1.2],
        zoom_range=[0.8,1.2],
    )

    val_generator = test_preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(36,36),
        color_mode='grayscale',
        shuffle=False,
        batch_size=batch_size
    )

    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=(36,36),
        color_mode='grayscale',
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, test_generator

def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1): # split the data for training 

    folders = os.listdir(path_to_data)

    for folder in folders:

        print("Folder: ", folder)

        full_path = os.path.join(path_to_data,folder)
        images_paths = glob.glob(os.path.join(full_path,'*.png'))

        x_train, x_val = train_test_split(images_paths, test_size=split_size)

        for x in x_train:

            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
            
            shutil.copy(x, path_to_folder)
        
        for x in x_val:

            path_to_folder = os.path.join(path_to_save_val, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
            
            shutil.copy(x, path_to_folder)

def create_border(path_to_data, size=2): # Create borders on images in a folder
    folders = os.listdir(path_to_data)
    
    for folder in folders:
        print("Folder: ", folder)

        full_path = os.path.join(path_to_data,folder) # Creates folder path for each folder
        
        for file_name in glob.glob(os.path.join(full_path,'*.png')):
            img = Image.open(file_name)
            color = "white"
            border = (size, size, size, size)
            new_img = ImageOps.expand(img, border=border, fill=color)
            new_img.save(file_name)

def ajust_images(path_to_raw, path_to_done): # Ajust any .png file to the network format
    for file_name in glob.glob(os.path.join(path_to_raw,'*.png')):
        img = Image.open(file_name)
        img = ImageOps.grayscale(img)
        img = img.resize((36,36), Image.ANTIALIAS)

        img_name = file_name.split("\\",-1)[-1]
        save_name = path_to_done + "\\" + img_name

        img.save(save_name)

def clean_trial_folder(path_to_folder): # Delete all .png's in a folder
    for file_name in glob.glob(os.path.join(path_to_folder,'*.png')):
        os.remove(file_name)

if __name__=='__main__':

    AJUST = True
    BORDER = False
    SPLIT = False

    if AJUST:

        path_to_raw = "C:\\Users\\Arthur\\Desktop\\Old"
        path_to_done = "C:\\Users\\Arthur\\Desktop\\New"

        ajust_images(path_to_raw,path_to_done)

    elif BORDER:
        path_to_data = "E:\\KanjiDataset\\FinalDataset"
        path_to_save_train_raw = "E:\\KanjiDataset\\Organized\\TrainRaw"
        path_to_save_test = "E:\\KanjiDataset\\Organized\\Test"
        
        print("Test - Start")
        split_data(path_to_data, path_to_save_train=path_to_save_train_raw, path_to_save_val=path_to_save_test, split_size=0.1)
        print("Test - Done")

        path_to_save_train = "E:\\KanjiDataset\\Organized\\Train"
        path_to_save_val = "E:\\KanjiDataset\\Organized\\Val"
        
        print("Val - Start")
        split_data(path_to_save_train_raw, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val, split_size=0.1)
        print("Val - Done")

    elif SPLIT:
        path_to_save_test = "E:\\KanjiDataset\\Organized\\Test"
        path_to_save_train = "E:\\KanjiDataset\\Organized\\Train"
        path_to_save_val = "E:\\KanjiDataset\\Organized\\Val"

        print("Borders - Test")
        create_border(path_to_save_test, size=4)
        print("Borders - Train")
        create_border(path_to_save_train, size=4)
        print("Borders - Val")
        create_border(path_to_save_val, size=4)
        print("Borders - Done")

        

        
        