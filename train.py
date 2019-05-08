# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:37:08 2019

@author: Yuan
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import net as mynet
import multibox
train_log_dir = 'logs\\train\\'
val_log_dir = 'logs\\val\\'
labeled_image_dir = 'labeled images\\'

#load tfrecord
def load_tfrecord(shuffle, path, batch_size):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([path],num_epochs=None) #读入流中
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                                   "label": tf.VarLenFeature(tf.int64),
                                                   "image" : tf.FixedLenFeature([], tf.string)
                                                   }
                                           )  #取出包含image和label的feature对象
        image = tf.decode_raw(features["image"],tf.uint8)
        image = tf.reshape(image,[300,300,3])  #H,W,C
        label = features["label"]
        if shuffle:
            image, label = tf.train.shuffle_batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=batch_size,
                                                  capacity=5,
                                                  min_after_dequeue=0)
        else:
            image, label = tf.train.batch([image, label],
                                          batch_size=batch_size,
                                          num_threads=batch_size,
                                          capacity=2000)
        image = tf.cast(image, tf.float32)
        return image, label


MAX_STEP= 1
BATCH_SIZE = 2
IMG_W = 300
IMG_H = 300
NUM_CLASSES = 1
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
DECAY_STEPS = 100

feature_layers=['ssd1', 'ssd2', 'ssd3', 'ssd4', 'ssd5', 'ssd6']
feature_shapes=[38,19,10,5,3,1]

tf.reset_default_graph()
tf.set_random_seed(233)

with tf.name_scope('Input'):
    image, label = load_tfrecord(shuffle=False, path="train_test.tfrecords", batch_size=BATCH_SIZE)
    image/=255.0
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])

#end_points = net.VGG16(x, base='VGG')
end_points, net = mynet.VGG16_Small(x, NUM_CLASSES, is_pretrain=True, usage='SSD')
end_points, net, predictions, localisations, logits = mynet.SSD(end_points)

class_prediction,location_prediction = multibox.prediction(end_points,
                                                           feature_layers,NUM_CLASSES)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
   
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            train_image, train_label = sess.run([image,label])#在会话中取出image和label
            cp = sess.run([class_prediction], feed_dict={x:train_image})
            
            
            
            print(type(cp[0][0]))
# =============================================================================
#             print(class_prediction[0].get_shape())
#             print(location_prediction[0].get_shape())
# =============================================================================
            
        for layer in feature_layers:
            print(layer, end_points[layer].get_shape())
            
            
            
            
# =============================================================================
#                 # save labeled images for check
#                 train_image = train_image.astype(np.uint8)
#                 new_im = Image.fromarray(train_image[0])
#                 draw = ImageDraw.Draw(new_im)
#                 for i in range(len(train_image)):
#                     for j in range(train_label[1][0]):
#                         draw.rectangle((train_label[1][4*j+1],train_label[1][4*j+2]
#                         , train_label[1][4*j+1]+train_label[1][4*j+4],
#                            train_label[1][4*j+2]+train_label[1][4*j+3]), outline='red')
#                 new_im.save(labeled_image_dir+str(step)+".jpg")
# =============================================================================
            
            
            
    except tf.errors.OutOfRangeError:
        print('Done reading')
    finally:
        coord.request_stop()
    coord.request_stop()
    coord.join(threads)

graph_def=tf.get_default_graph().as_graph_def()
_, wts_nd = mynet.showGraphNodes(graph_def,const_only=False)
    
print(len(train_image))
print(type(train_image[0][0][0][0]))
train_image_uint8 = (train_image*255).astype(np.uint8)
print(type(train_image_uint8[0][0][0][0]))
plt.imshow(train_image_uint8[0]) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()

