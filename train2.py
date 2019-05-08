# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:37:08 2019

@author: Yuan
"""
import os
import tensorflow as tf 
import cv2
import matplotlib.pyplot as plt
from PIL import Image
#load tfrecord
def load_tfrecord(is_train, path, batch_size=5):
    
    filename_queue = tf.train.string_input_producer([path],num_epochs=None) #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                               "label": tf.FixedLenFeature([], tf.int64),
                                               "image" : tf.FixedLenFeature([], tf.string)
                                               }
                                       )  #取出包含image和label的feature对象
    image = tf.decode_raw(features["image"],tf.uint8)
    image = tf.reshape(image,[32,32,3])
    label = features["label"]
    if is_train:
        image, label = tf.train.shuffle_batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=10,
                                              capacity=5,
                                              min_after_dequeue=0)
    else:
        image, label = tf.train.batch([image, label],
                                      batch_size=batch_size,
                                      num_threads=batch_size,
                                      capacity=2000)
    return image, label

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
MAX_STEP= 100
BATCH_SIZE = 5
IMG_W = 32
IMG_H = 32
N_CLASSES = 10

def train():
    tf.reset_default_graph()
    with tf.name_scope('input'):
        image, label = load_tfrecord(is_train=True, path="train.tfrecords", batch_size=BATCH_SIZE)
        timage, tlabel = load_tfrecord(is_train=False, path="train.tfrecords")
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int64, shape=[BATCH_SIZE, N_CLASSES]) 
    with tf.Session() as sess:
# =============================================================================
#         init_op = tf.global_variables_initializer()
#         sess.run(init_op)
# =============================================================================
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        
        try:
            for step in range(MAX_STEP):
                if coord.should_stop():
                    break
                example, l = sess.run([image,label])#在会话中取出image和label
                print(example.shape, l)
        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()
        coord.request_stop()
        coord.join(threads)
    plt.imshow(example[0]) # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.show()

train()
# =============================================================================
# load_tfrecord(is_train=True, MAX_STEP=100, path="test.tfrecords")
# =============================================================================
