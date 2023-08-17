import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Activation, Dropout
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.models import Sequential
import keras  
from keras.applications.vgg16 import VGG16
#from  keras.applications import VGG16, ResNet50
from keras import backend as K
from keras import optimizers
import os
import numpy as np
import errno
from matplotlib import pyplot as plt
import time



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        

        shutil.copy("C:\\Users\\arvind gowda\\Desktop\\Brain_tumor_vgg16\\test\\"+fileName, dst)
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        model=load_model('vgg.weights.best.hdf5')
        path='C:\\Users\\arvind gowda\\Desktop\\Brain_tumor_vgg16\\static\\images\\'+fileName
    ##    MODEL_NAME='keras_model.h5'
        
        model_out = (path,model)
        img = load_img(path,target_size=(224,224))
        plt.imshow(img)
       #img = load_img(path,target_size=(img_size,img_size))
        i = img_to_array(img)
       #im = preprocess_input(i)
        img = np.expand_dims(img,axis=0)
        model_out= model.predict(img)
        print(model_out)
        

        if np.argmax(model_out) == 0:
            str_label = "Brain Tumor "
           
            
            
        elif np.argmax(model_out) == 1:
            str_label  = "Normal"
            
                
            
                            

        return render_template('userlog.html', status=str_label,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName)
        
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
