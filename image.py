# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:58:54 2019

@author: Matej Pavlović
"""
import numpy as np
from PIL import Image

from keras.models import load_model
model = load_model('digits.h5')

im = Image.open("webcam-toy-photo2.jpg")

im = im.rotate(180)

im = im.crop((100,100,300,300))
im = im.resize((28,28))
im = im.convert("LA")

im = np.array(im)[:,:,0]/255

im = im.reshape((1,28,28,1))

pred = model.predict(im).argmax()