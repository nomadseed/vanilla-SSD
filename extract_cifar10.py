import numpy as np
from PIL import Image
import pickle
import os
import shutil

CHANNEL = 3
WIDTH = 32
HEIGHT = 32
data = []
labels=[]
classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
data_path = "D:\\dataset\\cifar-10-batches-py\\"
train_path = "D:\\dataset\\cifar-10-batches-py\\train_jpeg\\"
test_path = "D:\\dataset\\cifar-10-batches-py\\test_jpeg\\"
#parse train images
for i in range(5):
    with open(data_path + "data_batch_"+ str(i+1),mode='rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        data+= list(data_dict[b'data'])
        labels+= list(data_dict[b'labels'])

img =  np.reshape(data,[-1,CHANNEL, WIDTH, HEIGHT])

if not os.path.exists(train_path):
    os.makedirs(train_path)
for i in range(img.shape[0]):

    r = img[i][0]
    g = img[i][1]
    b = img[i][2]

    ir = Image.fromarray(r)
    ig = Image.fromarray(g)
    ib = Image.fromarray(b)
    rgb = Image.merge("RGB", (ir, ig, ib))

    name = "img-" + str(i) +"-"+ classification[labels[i]]+ ".png"
    rgb.save(train_path + name, "PNG")
#parse test images
data = []
labels=[]
with open(data_path + "test_batch",mode='rb') as file:
    data_dict = pickle.load(file, encoding='bytes')
    data+= list(data_dict[b'data'])
    labels+= list(data_dict[b'labels'])
img =  np.reshape(data,[-1,CHANNEL, WIDTH, HEIGHT])

  
if not os.path.exists(test_path):
    os.makedirs(test_path)
for i in range(img.shape[0]):

    r = img[i][0]
    g = img[i][1]
    b = img[i][2]

    ir = Image.fromarray(r)
    ig = Image.fromarray(g)
    ib = Image.fromarray(b)
    rgb = Image.merge("RGB", (ir, ig, ib))

    name = "img-" + str(i) +"-"+ classification[labels[i]]+ ".png"
    rgb.save(test_path + name, "PNG")

#create folders for 10 classes
for clas in classification:
    if not os.path.exists(train_path+clas):
        os.mkdir(train_path+clas)
    if not os.path.exists(test_path+clas):
        os.mkdir(test_path+clas)
#move train images to corresponding folders
file_list = os.listdir(train_path)
for name in file_list:
    if len(name.split("-")) != 1:
        clas = name.split("-")[2].split(".")[0]
        shutil.move(train_path+name,train_path+clas+"\\"+name)
#move test images to corresponding folders 
file_list = os.listdir(test_path)
for name in file_list:
    if len(name.split("-")) != 1:
        clas = name.split("-")[2].split(".")[0]
        shutil.move(test_path+name,test_path+clas+"\\"+name)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    