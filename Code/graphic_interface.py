from cgitb import text
from ipaddress import collapse_addresses
import tkinter as tk
import threading as th
import tensorflow as tf
import webbrowser
import subprocess
import time
from kanji_info import KanjiInfo
from tkinter import ttk, Label, Button
from PIL import ImageTk, Image, ImageGrab
from my_predictor import create_dictionary, clean_trial_folder

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set Initial Parameters
        self.title('Kanji Detector')
        self.geometry('850x450')
        self.iconbitmap('./Resources/Icon.ico')
        #self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Style Settings
        style = ttk.Style()
        style.theme_use('xpnative')
        style.configure('W.TButton', font = ('arial', 10, 'bold'))

        # Get Dictionary and img folder
        name  = "./Resources/KanjiTable.csv"
        self.kanjiDict = create_dictionary(name)
        self.folder_path = './Temp/Images'
        self.Kanji_image = Image.open('./Resources/default_raw.png')

        # create threads
        self.tImage = th.Thread(target=self.updateImage)
        self.activeT = False

        # loading the model
        self.modelo = None
        open_net = th.Thread(target=self.openNet)
        open_net.start()

        # create history
        self.Kanji = []
        
        # Default Info
        info_list = self.kanjiDict[185]

        # Set frames
        Inframe = ttk.Frame(self)
        Inframe.grid(column=0, row=0)

        Outframe = ttk.LabelFrame(self, text="Resultado")
        Outframe.grid(column=1, row=0, padx=10, pady=10)

        self.historyframe = ttk.LabelFrame(self, text="Historico")
        self.historyframe.grid(column=0, row=3, columnspan=2, padx=10, pady=10)

        # Set labels
        Label(Inframe, text="Detecção de Kanji", font=("Arial", 15), width=17).grid(column=0, row=0, pady=5)

        frame_resposta = ttk.LabelFrame(Outframe, text="Resposta:")
        frame_resposta.grid(column=0, row=0, ipadx=10, ipady=10, padx=10, pady=10)
        self.label_kanji = Label(frame_resposta, text=info_list[0], font="Arial 35 bold")
        self.label_kanji.pack()

        frame_radical = ttk.LabelFrame(Outframe, text="Radical:")
        frame_radical.grid(column=0, row=1, ipadx=10, ipady=10, padx=10, pady=10)

        self.label_radical = Label(frame_radical, text=info_list[1], font="Arial 30")
        self.label_radical.pack()

        frame_meaning = ttk.LabelFrame(Outframe, text="Siguinificado(en/pt):")
        frame_meaning.grid(column=2, row=0, columnspan=3, ipadx=10, ipady=10, padx=5, pady=5)
        self.label_meaning = Label(frame_meaning, text=info_list[3] + " / " + info_list[4], font="Arial 14 bold", wraplength=350, justify="left")
        self.label_meaning.pack()

        frame_kana = ttk.LabelFrame(Outframe, text="Kana Reading:")
        frame_kana.grid(column=2, row=1, columnspan=3, ipadx=10, ipady=10, padx=5, pady=5)
        self.label_kana_reading = Label(frame_kana, text=info_list[5], font="Arial 14 bold", wraplength=450, justify="left")
        self.label_kana_reading.pack()

        frame_romanji = ttk.LabelFrame(Outframe, text="Romanji Reading:")
        frame_romanji.grid(column=2, row=2, columnspan=3, ipadx=10, ipady=10, padx=5, pady=5)
        self.label_romanji_reading = Label(frame_romanji, text=info_list[6], font="Arial 14 bold", wraplength=450, justify="left")
        self.label_romanji_reading.pack()

        # Set buttons
        self.GetImageB = ttk.Button(Inframe, text="Loading...", style = 'W.TButton', command=self.GetImages, state="disabled", cursor= "hand2")
        self.GetImageB.grid(column=0, row=2, ipady=5, ipadx=55, padx=15, pady=3, sticky="ew")

        self.PredictB = ttk.Button(Inframe, text="Loading...", style = 'W.TButton', command=self.PredicImage, state="disabled", cursor= "hand2")
        self.PredictB.grid(column=0, row=3, ipady=5, ipadx=55, padx=15, pady=3, sticky="ew")

        self.GoToPageB = ttk.Button(Inframe, text="Loading...", style = 'W.TButton', command=self.goToWiki, state="disabled", cursor= "hand2")
        self.GoToPageB.grid(column=0, row=4, ipady=5, ipadx=55, padx=15, pady=3, sticky="ew")

        # default image
        self.img_raw = ImageTk.PhotoImage(self.Kanji_image.resize((180,180), Image.ANTIALIAS))
        self.raw_img = Label(Inframe, text="image", image=self.img_raw, borderwidth=3, relief="solid")
        self.raw_img.grid(column=0, row=1, padx=5, pady=10)

        self.img_done = ImageTk.PhotoImage(Image.open('./Resources/default_done.png').resize((36,36), Image.ANTIALIAS))
        self.done_img = Label(Outframe, text="image", image=self.img_done, borderwidth=2, relief="solid")
        self.done_img.grid(column=0, row=2, padx=10, pady=10)

    def writeHistory(self):
        pass

    def on_closing(self):
        if self.tImage.is_alive():
            self.activeT = False
            self.tImage.join()
        clean_trial_folder(self.folder_path) # Lembra de mudar de volta
        self.destroy()

    def openNet(self): 
        self.modelo = tf.keras.models.load_model('./Models/original')
        self.PredicImage()
        self.GetImageB.config(text="Get image",state="enable")
        self.PredictB.config(text="PREDICT!", state="enable")
        self.GoToPageB.config(text="Go To Wiki",state="enable")   

    def GetImages(self):
        if not self.tImage.is_alive():
            self.Information = ttk.Label(self, text="Você pode tambem usar o comando [Windows + Shift + S] Para recaptura a imagem", font="Arial 13 bold")
            self.Information.grid(column=0, row=1, columnspan=2, pady=3, padx=20, sticky="ew")
            self.GetImageB.config(text="Scan a New Image")
            subprocess.run(["explorer.exe", "ms-screenclip:"])
            self.activeT = True
            self.tImage.start()
        else:
            subprocess.run(["explorer.exe", "ms-screenclip:"])

    def updateImage(self):
        while(self.activeT):
            try:
                self.Kanji_image = ImageGrab.grabclipboard().resize((180,180), Image.ANTIALIAS)
                self.img_raw = ImageTk.PhotoImage(self.Kanji_image)
                self.raw_img.config(image=self.img_raw) #= Label(self, text="image", image=self.img_raw, borderwidth=3, relief="solid")
            except AttributeError as error:
                self.img_raw = ImageTk.PhotoImage(Image.open('./Resources/INF_raw.png').resize((180,180), Image.ANTIALIAS))
                self.raw_img.config(image=self.img_raw) #= Label(self, text="image", image=self.img_raw, borderwidth=3, relief="solid")
            except Exception as exception:
                print("\nA Unexpected Exception Has Appeared")
                return None
            
            time.sleep(1)      

    def PredicImage(self):

        start = time.time()

        kanji = KanjiInfo(self.modelo, self.kanjiDict, self.Kanji_image, self.folder_path)
        self.Kanji.insert(0, kanji)

        end = time.time()

        #self.writeHistory()

        # set labels
        self.label_kanji.config(text=kanji.kanji)
        self.label_radical.config(text=kanji.radical)
        self.label_meaning.config(text=kanji.meaning_en + " | " + kanji.meaning_pt)
        self.label_kana_reading.config(text=kanji.kana_reading)
        self.label_romanji_reading.config(text=kanji.romanji_reading)

        #set image
        img = Image.open(kanji.img_net)
        self.img_done = ImageTk.PhotoImage(img)
        self.done_img.config(image=self.img_done)

        kanji.print_info()
        print("Tempo de Execução: ")
        print(end - start)

    def goToWiki(self):
        if self.Kanji:
            webbrowser.open("https://jisho.org/search/%23kanji " + self.Kanji[0].kanji)
        else:
            webbrowser.open("https://jisho.org/search/%23kanji 語")

if __name__ == "__main__":
    app = App()
    app.mainloop()