# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 23:08:07 2019

@author: Yuan
"""

import os
import tensorflow as tf 
import numpy as np
from PIL import Image
import json

#create tfrecord

'''
format of the value:
    var list: x, y, height, width ...
'''
def write_tfrecord(is_train):
    if is_train:
        writer = tf.python_io.TFRecordWriter("train.tfrecords", options=None)
    else:
        writer = tf.python_io.TFRecordWriter("test.tfrecords", options=None)
    '''
        option：TFRecordOptions对象，定义TFRecord文件保存的压缩格式；
        有三种文件压缩格式可选，分别为TFRecordCompressionType.ZLIB、TFRecordCompressionType.GZIP
        以及TFRecordCompressionType.NONE，默认为最后一种，即不做任何压缩，定义方法如下：
        option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    '''
    i = 0
    if is_train:
        img_folder_path = "D:\\dataset\\mySDD\\train\\"
        label_path = "D:\\dataset\\mySDD\\label.json"
    else:
        img_folder_path = "D:\\dataset\\SDDtest\\"
    with open(label_path, 'r') as infile:
        label = json.load(infile)

    for img_name in os.listdir(img_folder_path):
        i = i + 1
        img_path = os.path.join(img_folder_path,img_name)
        img = Image.open(img_path)
        img = np.array(img)
        img_raw = img.tobytes()
        total_num = 0
        for name in label:
            if name == img_name:
                total_num = len(label[name]["annotations"])
                label_list = [total_num]
                for num in range(total_num):
                    label_list.append(label[name]["annotations"][num]["x"])
                    label_list.append(label[name]["annotations"][num]["y"])
                    label_list.append(label[name]["annotations"][num]["height"])
                    label_list.append(label[name]["annotations"][num]["width"])
                print(label_list)
                
        feature_internal = {
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                "label":tf.train.Feature(int64_list=tf.train.Int64List(value=label_list))
                }
        features_extern = tf.train.Features(feature=feature_internal)
        example = tf.train.Example(features=features_extern)
        example_str = example.SerializeToString()
        writer.write(example_str)  #序列化为字符串

        print(str(i)+ " images processed")
        
    writer.close()
    if is_train:
        print("finished generating train.tfrecords, now beging generating test.rfrecords")
    else:
        print("finished generating test.tfrecords")


write_tfrecord(is_train=True)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    