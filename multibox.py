# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 00:21:55 2019

@author: Yuan
"""
import tensorflow as tf
slim = tf.contrib.slim

def prediction(end_points,feature_layers,NUM_CLASSES):
    with tf.variable_scope('Prediction'):
        class_prediction_list = []
        location_prediction_list = []
        
        
        n_channels = end_points['ssd1'].get_shape().as_list()[-1]
        l2_norm = tf.nn.l2_normalize(end_points['ssd1'],axis=[3],epsilon=1e-12)
        with tf.variable_scope('L2Normalization'):
            gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                    initializer=tf.constant_initializer(20),
                                    trainable=True)
        end_points['ssd1_L2Norm'] =  l2_norm * gamma

        print(feature_layers)
        for layer in feature_layers:
            class_prediction = tf.nn.softmax(slim.conv2d(end_points[layer], NUM_CLASSES+1,
                                            [3, 3], activation_fn=None,
                                            stride=1),axis=-1,
                                            name='cls_predict_{}'.format(layer))
            class_prediction_list.append(class_prediction)
            location_prediction = slim.conv2d(end_points[layer], 4, [3, 3],
                                              activation_fn=None, stride=1)
            location_prediction=tf.identity(location_prediction,'loc_predict_{}'.format(layer))
            location_prediction_list.append(location_prediction)
        return class_prediction_list, location_prediction_list