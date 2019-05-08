# -*- coding: utf-8 -*-
import tensorflow as tf
import multibox
from tensorflow.python.framework import tensor_util
slim = tf.contrib.slim



def SSD(end_points, num_classes=1, is_training=True, netlist=None, anchorlist=None):
    # setups
    
    normalization=True # for multibox layer
    dropout_keep_prob=0.5
    anchor_sizes = [(21., 45.),
                    (45., 99.),
                    (99., 153.),
                    (153., 207.),
                    (207., 261.),
                    (261., 315.)] # base size for anchors
    anchor_ratios=[1, 0.5, 2] # anchor ratios for base anchors
    
    # SSD feature extractor layers
    with tf.variable_scope('SSD_extractor'):
        if netlist is None:
            #use the last layer of the endpoints as input, and do conv2d for the 
            #feature maps
            featurelayers=['ssd1','ssd2','ssd3','ssd4','ssd5','ssd6']
            # Block 1.
            net = slim.conv2d(end_points['block5'], 1024, [3, 3], stride=1, rate=6, scope='ssd1', trainable=is_training)
            end_points['ssd1'] = net
            net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
            # Block 2.
            net = slim.conv2d(net, 1024, [1, 1], stride=1, scope='ssd2', trainable=is_training)
            end_points['ssd2'] = net
            net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
            # Block 3.
            net = slim.conv2d(net, 256, [1, 1], stride=1, scope='ssd3_1', trainable=is_training)
            net = tf.pad(net,paddings=[[0,0],[1,1],[1,1],[0,0]],mode='CONSTANT')
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='ssd3_2', padding='VALID', trainable=is_training)
            end_points['ssd3'] = net
            # Block 4.
            net = slim.conv2d(net, 128, [1, 1], stride=1, scope='ssd4_1', trainable=is_training)
            net = tf.pad(net,paddings=[[0,0],[1,1],[1,1],[0,0]],mode='CONSTANT')
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='ssd4_2', padding='VALID', trainable=is_training)
            end_points['ssd4'] = net
            # Block 5.
            net = slim.conv2d(net, 128, [1, 1], stride=1, scope='ssd5_1', trainable=is_training)
            net = slim.conv2d(net, 256, [3, 3], stride=1, scope='ssd5_2', padding='VALID', trainable=is_training)
            end_points['ssd5'] = net
            # Block 6.
            net = slim.conv2d(net, 128, [1, 1], stride=1, scope='ssd6_1', trainable=is_training)
            net = slim.conv2d(net, 256, [3, 3], stride=1, scope='ssd6_2', padding='VALID', trainable=is_training)
            end_points['ssd6'] = net
        
        else:
            featurelayers=[]
            raise IOError('customized SSD structure not supported yet.')
# =============================================================================
#         
#         # Prediction and localisations layers.
#         predictions = []
#         logits = []
#         localisations = []
#         for i, layer in enumerate(featurelayers):
#             with tf.variable_scope(layer + '_box'):
#                 p, l = ssd_multibox_layer(end_points[layer],
#                                           num_classes,
#                                           anchor_sizes[i],
#                                           anchor_ratios,
#                                           normalization=normalization)
#             predictions.append(slim.softmax(p))
#             logits.append(p)
#             localisations.append(l)
# =============================================================================
            
        
    return end_points, net#, predictions, localisations, logits

def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios,
                       normalization=True):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization:
        net = tf.nn.l2_normalize(net)
    # Number of anchors.
    num_anchors = len(sizes) * len(ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='loc')
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred)[:-1]+[num_anchors, 4])
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='cls')
    #print('cls {}'.format(cls_pred.get_shape().is_fully_defined()))
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred)[:-1]+[num_anchors, num_classes])
    return cls_pred, loc_pred

def tensor_shape(x):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        raise ValueError('SSD extractor not fully \
                         built,failed building tensor {}'.format(x.name))

def VGG16_Small(net, n_classes, is_training=True, usage='VGG16'):
    """
    this is a lite version of VGG16, with small FC layers and has stride 2 pooling
    
    """
    end_points = {}
    with tf.variable_scope('vgg16'):
        # block 1
        net=slim.conv2d(net, 64, [3, 3], scope='conv1_1', trainable=is_training)
        net=slim.conv2d(net, 64, [3, 3], scope='conv1_2', trainable=is_training)
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')
        # block 2
        net=slim.conv2d(net, 128, [3, 3], stride=1, scope='conv2_1', trainable=is_training)
        net=slim.conv2d(net, 128, [3, 3], scope='conv2_2', trainable=is_training)
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')
        # block 3
        net=slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3_1', trainable=is_training)
        net=slim.conv2d(net, 256, [3, 3], scope='conv3_2', trainable=is_training)
        net=slim.conv2d(net, 256, [1, 1], scope='conv3_3', trainable=is_training)
        end_points['block3'] = net
        net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pool3')
        # Block 4.
        net=slim.conv2d(net, 512, [3, 3], stride=1, scope='conv4_1', trainable=is_training)
        net=slim.conv2d(net, 512, [3, 3], scope='conv4_2', trainable=is_training)
        net=slim.conv2d(net, 512, [1, 1], scope='conv4_3', trainable=is_training)
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool4')
        # Block 5.
        net=slim.conv2d(net, 512, [3, 3], stride=1, scope='conv5_1', trainable=is_training)
        net=slim.conv2d(net, 512, [3, 3], scope='conv5_2', trainable=is_training)
        net=slim.conv2d(net, 512, [1, 1], scope='conv5_3', trainable=is_training)
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='pool5')

        if usage=='VGG16':
            # for classification
            net=slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, 
                                     scope='fc1')
            net=slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, 
                                     scope='fc2')
            net=slim.fully_connected(net, 1024, activation_fn=tf.nn.relu,
                                     scope='fc3')
            net==slim.fully_connected(net,n_classes,
                                      activation_fn=tf.nn.softmax,scope='fc4')
            end_points['output'] = net
            return end_points, net
        elif 'SSD' in usage:
            # for SSD-based detection      
            return end_points, net
        else:
            raise ValueError('the usage "{}" in not included'.format(usage))


def CNN(x, n_classes, is_pretrain=True):
    
    with tf.name_scope('CNN'):

        x = conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True)   
        x = conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True)
        with tf.name_scope('pool1'):    
            x = pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            
        x = conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True)    
        x = conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True)
        with tf.name_scope('pool2'):    
            x = pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
         
            

        x = conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True)
        x = conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True)
        x = conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True)
        with tf.name_scope('pool3'):
            x = pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

        x = FC_layer('fc4', x, out_nodes=1000)        
        with tf.name_scope('batch_norm1'):
            x = batch_norm(x)           
        x = FC_layer('fc5', x, out_nodes=1000)        
        with tf.name_scope('batch_norm2'):
            x = batch_norm(x)            
        x = FC_layer('fc6', x, out_nodes=n_classes)
    
        return x




def conv(layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers. 
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer()) # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x

def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
    '''Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    '''
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x

def FC_layer(layer_name, x, out_nodes):
    '''Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size]) # flatten into 1D
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        return x

def batch_norm(x):
    '''Batch normlization(I didn't include the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x

def showGraphNodes(od_graph_def, const_only=True):
    
    graph_nodes=[n for n in od_graph_def.node]
    wts_nd={}
    if const_only:
        wts = [n for n in graph_nodes if n.op=='Const']
        for n in wts:
            wts_nd[n.name]=tensor_util.MakeNdarray(n.attr['value'].tensor)
    else:
        for n in graph_nodes:
            if n.op=='Const':
                wts_nd[n.name]=tensor_util.MakeNdarray(n.attr['value'].tensor)
            else:
                wts_nd[n.name]=n.attr['value'].tensor
    
    
    return graph_nodes, wts_nd

if __name__=='__main__':
    extern_lib='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/tf-object-detection-api/research/list_graph.py'
    n_classes=1
    
    #setup env
    tf.reset_default_graph() 
    
    #define input
    x=tf.random_normal([1,300,300,1],dtype=tf.float32,name='input')
    end_points, net = VGG16_Small(x, n_classes, is_training=True, usage='SSD')
    model = SSD(end_points, num_classes=n_classes, is_training=True)
    init=tf.global_variables_initializer()
    
    #check graph
    graph_def=tf.get_default_graph().as_graph_def()
    _, wts_nd = showGraphNodes(graph_def,const_only=False)
    node_names=[n for n in wts_nd]
    
    with tf.Session() as sess:
        sess.run(init)
        end_points, net= sess.run(model)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                           tf.get_default_graph().as_graph_def(),
                           node_names)   
    #show frozen graph
    _, wts_nd = showGraphNodes(graph_def,const_only=True)
    
    