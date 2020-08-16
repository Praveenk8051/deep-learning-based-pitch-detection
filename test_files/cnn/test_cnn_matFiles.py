import tensorflow as tf
import numpy as np
import os
import scipy.io

saveNetworkoutput=r'**Enter the path**'
name='matfile'
mat = scipy.io.loadmat(r'**Enter the path to mat file**')
gray  = mat['a']
gray=np.expand_dims(gray,axis=0)
gray=np.expand_dims(gray,axis=3)

with tf.Session() as sess_CnnNet:    

    saver = tf.train.import_meta_graph(r'**Enter the path of the .meta file**')
    saver.restore(sess_CnnNet,tf.train.latest_checkpoint(r'**Emter the path where the files(checkpoint, meta and .data) is present**'))
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("Placeholder:0") #label
    w2 = graph.get_tensor_by_name("Placeholder_1:0") #input
    operation = graph.get_tensor_by_name("deconv1/deconv1/Relu:0")


    denoisedOutputs=sess_CnnNet.run(operation,{w2:gray})
    denoisedOutputs=np.squeeze(denoisedOutputs,axis=0)
    scipy.io.savemat(os.path.join(saveNetworkoutput,name+'.mat'),{'a':denoisedOutputs})