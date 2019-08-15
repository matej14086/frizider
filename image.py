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
while True:
    
    data = ser.readline()
    data = data.decode().split(" ")
    
    otpor = float(data[0])
    status = data[1][0]
    
    file.write(time.ctime() +" " + str(otpor) + "\n")
    
    
    

os.system("git add .")
os.system("git commit -m 'auto'")
os.system("git push")

im = Image.open("webcam-toy-photo4.jpg")

im = im.rotate(180)

im =  im.crop((100,200,350,450))
im = im.resize((28,28))
im =  im.convert("RGB")

im = np.array(im)[:,:,1]/255

im = im.reshape((1,28,28,1))

pred = model.predict(im).argmax()