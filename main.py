import time
from tkinter import *
from tkinter.filedialog import askopenfilename

from PIL import ImageTk, Image
import sqlite3
import os

root = Tk()
root.geometry('1366x768')
root.title("Intrusion")
canv = Canvas(root, width=1366, height=768, bg='white')
canv.grid(row=2, column=3)
img = Image.open('back.png')
photo = ImageTk.PhotoImage(img)
canv.create_image(1, 1, anchor=NW, image=photo)
File = StringVar()


def Load():
    os.system("python dataset.py")


def pre():
    os.system('python preprocessing.py')

def train():
    os.system('python train_ddos_model.py')
def pred():
    os.system('python ddos_gui_app.py')

def model():
    os.system('python ensmodel.py')

Button(root, text='Load Dataset', width=30, bg='yellow', fg='black', font=("bold", 12), command=Load).place(x=300,y=400)
Button(root, text='Data Preprocessing', width=30, bg='yellow', fg='black', font=("bold", 12), command=pre).place(x=300, y=450)
Button(root, text='Model Loading (Trained Ensemble)', width=30, bg='yellow', fg='black', font=("bold", 12), command=train).place(x=300, y=500)
Button(root, text='Ensemble Model', width=30, bg='yellow', fg='black', font=("bold", 12), command=model).place(x=300, y=550)
Button(root, text='DDoS Attack Detection', width=30, bg='yellow', fg='black', font=("bold", 12), command=pred).place(x=300, y=600)

root.mainloop()
