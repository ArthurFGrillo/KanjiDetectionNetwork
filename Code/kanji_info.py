from my_predictor import predict_with_model, create_image_from_clipboard, predict_with_model, create_dictionary
import tensorflow as tf
from PIL import Image, ImageOps
import os

class KanjiInfo():
    def __init__(self, model, dict, image, path):

        # Create images
        self.create_images(path, image)

        # predict !!!
        val = predict_with_model(model, self.img_net)

        # get Info
        info_list = dict[val]

        # set Values
        self.kanji = info_list[0]
        self.radical = info_list[1]
        self.grades = info_list[2]
        self.meaning_en = info_list[3]
        self.meaning_pt = info_list[4]
        self.kana_reading = info_list[5]
        self.romanji_reading = info_list[6]

    def create_images(self, save_path, image): 
        
        list = os.listdir(save_path)
        num_files = len(list)
        img_name = ((num_files//2)+1)
        path_raw = save_path + "/Raw{}.png".format(img_name)
        image.save(path_raw,'PNG') 

        edited_img = ImageOps.grayscale(image)
        edited_img = edited_img.resize((36,36), Image.ANTIALIAS)
        path_net = save_path + "/Net{}.png".format(img_name)
        edited_img.save(path_net,'PNG')

        self.img_raw = path_raw
        self.img_net = path_net        

    def print_info(self): # answer with a user friendly message
        print("\nDetected Kanji:", self.kanji, " Meaning:", self.meaning_en, "| Siginificado:", self.meaning_pt)
        print("It's pronunciation in Kana is:", self.kana_reading)
        print("It's pronunciation in Romanji is:", self.romanji_reading)


if __name__ == "__main__":
    # Geting the folder paths
    image_folder = './Temp/Images'

    # loading the model
    modelo = tf.keras.models.load_model('./Models')

    # Answering the dictionarys
    name  = "./Resources/KanjiTable.csv"
    kanjiDict = create_dictionary(name)

    #image
    img = Image.open('./Resources/default_raw.png')

    kanji = KanjiInfo(modelo, kanjiDict, img, image_folder)
    kanji.print_info()

