import tensorflow as tf
import numpy as np
import cv2
import os

numberofImages=50
saveNetworkoutput=r'**Enter the path**'
pickfiles=r'**Enter the path**'
num=0

for i in range(numberofImages):
    num=num+1
    names=("specInput%d" % (num))

    img=cv2.imread(os.path.join(pickfiles,names+ "." + 'png'))

    
    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=np.expand_dims(gray,axis=0)
    gray=np.expand_dims(gray,axis=3)
    
    with tf.Session() as sess_CnnNet:    

        saver = tf.train.import_meta_graph(r'**Enter the path of the .meta file**')
        saver.restore(sess_CnnNet,tf.train.latest_checkpoint(r'**Emter the path where the files(checkpoint, meta and .data) is present**'))
        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name("Placeholder:0") #label
        w2 = graph.get_tensor_by_name("Placeholder_1:0") #input
        operation = graph.get_tensor_by_name("deconv1/deconv1/Relu:0")


        denoisedOutput=sess_CnnNet.run(operation,{w2:gray})
        denoisedOutput=np.squeeze(denoisedOutput,axis=0)
        cv2.imwrite(os.path.join(saveNetworkoutput,names+ "." + 'png'),denoisedOutput)
