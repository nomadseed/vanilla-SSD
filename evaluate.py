# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:37:08 2019

@author: Yuan
"""

import tensorflow as tf
import numpy as np
import net
import tools
import math
train_log_dir = 'logs\\train\\'

#load tfrecord
def load_tfrecord(shuffle, path, batch_size):
    with tf.name_scope('input'):
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
    
        label = tf.one_hot(label, depth= N_CLASSES)
        label = tf.reshape(label, [BATCH_SIZE, N_CLASSES])
        return image, label

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

BATCH_SIZE = 100
N_CLASSES = 10
n_test = 10000

def evaluate():
    tf.reset_default_graph()
    val_image, val_label = load_tfrecord(shuffle=False, path="test.tfrecords", batch_size=BATCH_SIZE)
    
    logits = net.CNN(val_image, N_CLASSES, is_pretrain=True)
    correct = tools.num_correct_prediction(logits, val_label)
    saver = tf.train.Saver(tf.global_variables())
    
    with tf.Session() as sess:
# =============================================================================
#         tf.get_variable_scope().reuse_variables()
# =============================================================================
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(train_log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
            return

        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        
        try:
            print('\nEvaluating......')
            num_step = int(math.floor(n_test / BATCH_SIZE))
            num_sample = num_step*BATCH_SIZE
            step = 0
            total_correct = 0
            while step < num_step and not coord.should_stop():
                batch_correct = sess.run(correct)
                total_correct += np.sum(batch_correct)
                step += 1
            print('Total testing samples: %d' %num_sample)
            print('Total correct predictions: %d' %total_correct)
            print('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
            
        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()
        coord.request_stop()
        coord.join(threads)

eva = evaluate()

