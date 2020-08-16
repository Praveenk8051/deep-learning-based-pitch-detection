import tensorflow as tf
import pickle
import numpy as np
import os
import natsort
import scipy.io
import os.path

saveNetworkoutput=r'**Enter the path**'
batch_size=50
timeSteps=216
imagesPath=r'**Enter the path**'
imagesFilepath = []

for d in os.listdir(imagesPath):
    imagesFilepath.append(os.path.join(imagesPath, d))
  
imagesFilepath=natsort.natsorted(imagesFilepath,reverse=False)
trainInputs = imagesFilepath

def _read_py_function(inputs):
    f=open(inputs, 'rb')
    inputs_=pickle.load(f)
    f.close()
    return inputs_

def parse_function(inputs):
    casted_inputs= tf.cast(inputs,dtype=tf.float32)
    return casted_inputs

############################################TRAIN##################################################
dataset_trainInputs=tf.data.Dataset.from_tensor_slices(trainInputs)
dataset_trainInputs=dataset_trainInputs.map(lambda trainInputs: tf.py_func(_read_py_function, [trainInputs], [tf.uint8]))
dataset_trainInputs=dataset_trainInputs.map(parse_function)
dataset_trainInputs=dataset_trainInputs.repeat(count=1000)
dataset_trainInputs=dataset_trainInputs.batch(batch_size)
iterator_trainInputs=dataset_trainInputs.make_one_shot_iterator()
next_element_trainInputs=iterator_trainInputs.get_next()



with tf.Session() as sess_CnnNet:    
    
    saver=tf.train.import_meta_graph(r'Enter the path\9_256_64_54.meta')
    saver.restore(sess_CnnNet,tf.train.latest_checkpoint(r'Folder where files reside'))
    graph=tf.get_default_graph()
    w1=graph.get_tensor_by_name("batch_x:0") #label
    w2=graph.get_tensor_by_name("batch_y:0") #input
    w3=graph.get_tensor_by_name("Placeholder:0") #input
    operation=graph.get_tensor_by_name("fully_connected/Relu:0")
    midi_obj=sess_CnnNet.run(next_element_trainInputs)
    
    _current_state=np.zeros((2,2,batch_size,110))
    midi_values=sess_CnnNet.run(operation,{w1:midi_obj,w3:_current_state})

#One hot vector
for i in range(batch_size):
    for j in range(timeSteps):
            temp_midi=midi_values[i,j,:]
            temp_zeros=np.zeros((1,len(midi_values[i,j,:])))
            temp_zeros=np.squeeze(temp_zeros)
            posMaxValue=np.argmax(temp_midi)
            temp_zeros[posMaxValue]=1
            midi_values[i,j,:]=temp_zeros

#Reshape the midi dimension to image dimension
num=0
value=0
for i in range(batch_size):
    a=midi_values[i,:,:]
    a_=np.transpose(a)
    if value==0:
        data=a_
        value=1
    else:    
        data=np.concatenate((data,a_), axis=1)
        
    if (i+1)%1==0:
        num=num+1
        name="lstmNetworkoutput%d"%(num)
        print(data.shape)
        scipy.io.savemat(os.path.join(saveNetworkoutput,name+'.mat'),{'variable':data})
        value=0


