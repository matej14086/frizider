# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:58:54 2019

@author: Matej Pavlović
"""
import numpy as np
from PIL import Image
import os
import serial
import time


from keras.models import load_model
model = load_model('digits.h5')

ser = serial.Serial('COM8', 9600) 

file = open("data/OtporData.txt","a")
fileT = open("data/TempData.txt","a")
otpori = []
temps = []
num_files = len(os.listdir("data"))

ser.flush()
while True:
    
    data = ser.readline()
    data = data.decode().split(" ")
    
    otpor = float(data[0])
    status = data[1][0]
    
    file.write(time.ctime() +" " + str(otpor) + "\n")
    
    otpori.append(otpor)
    
    if num_files != len(os.listdir("data")):
        
        num_files = len(os.listdir("data"))
        
        files = os.listdir("data/")
        
        times = [os.path.getctime("data/" + i) for i in files]
        
        slika = files[np.argmax(times)]
        
        
        im = Image.open("data/" + slika)
        
        im = im.rotate(180)
        im =  im.crop((30,50,230,250))
        im = im.resize((28,28))
        im =  im.convert("RGB")
        im = np.array(im)[:,:,1]/255
        im = im.reshape((1,28,28,1))
        pred = model.predict(im).argmax()
                
        temps.append(pred)
        fileT.write(time.ctime() +" " + str(pred) + "\n")
        with open("README.md") as f:
            lines = f.readlines()
            
        temp = lines[0].split(" ")
        temp[2] = str(pred) + "\n"
        lines[0] = " ".join(temp)
        lines[1] = str(np.mean(otpori)) + "\n"
        lines[2] = status
        
        otpori = []

        with open("README.md", "w") as f:
            f.writelines(lines)
        
        print("Pushing to git")

        os.system("git add .")
        os.system("git commit -m 'auto'")
        os.system("git push")
