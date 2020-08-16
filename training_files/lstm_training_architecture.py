import tensorflow as tf
import numpy as np
import pickle
import natsort
import os 

#define parameters
number_of_layers = 2
numberEpochs=5
state_size=128
batch_size=1
timeSteps=216

#Enter the path where log files needs to be saved
logs_path=r"**Enter the path**"


#Enter the paths where datasets reside
labelsPath=r"**Enter the path**"
inputsPath=r"**Enter the path**"


def createfilepaths(labelsPath,inputsPath):

    labelsList=[]
    midiList=[]
    for d in os.listdir(labelsPath):
        labelsList.append(os.path.join(labelsPath, d))

    for d in os.listdir(inputsPath):
        midiList.append(os.path.join(inputsPath, d))    
    
    
    labelsList=natsort.natsorted(labelsList,reverse=False)
    midiList=natsort.natsorted(midiList,reverse=False)

    return labelsList,midiList

labelsList,midiList=createfilepaths(labelsPath,inputsPath)

trainLabels=labelsList[:int(len(labelsList)*0.75)]
trainInputs=midiList[:int(len(midiList)*0.75)]
    

validationLabels=labelsList[-int(len(labelsList)*0.25):]
validationInputs=midiList[-int(len(midiList)*0.25):]


#Convert pickle data into normal format data
def _read_py_function(inputs):
    f = open(inputs, 'rb')
    inputs_ = pickle.load(f)
    f.close()
    return inputs_

#unsigned integer value to float 32
def parse_function(inputs):
    casted_inputs= tf.cast(inputs,dtype=tf.float32)
    return casted_inputs

############################################TRAIN##################################################
dataset_trainInputs = tf.data.Dataset.from_tensor_slices(trainInputs)
dataset_trainInputs = dataset_trainInputs.map(lambda trainInputs: tf.py_func(_read_py_function, [trainInputs], [tf.uint8]))
dataset_trainInputs= dataset_trainInputs.map(parse_function)
dataset_trainInputs = dataset_trainInputs.repeat(count=1000)
dataset_trainInputs = dataset_trainInputs.batch(batch_size)
iterator_trainInputs = dataset_trainInputs.make_one_shot_iterator()
next_element_trainInputs= iterator_trainInputs.get_next()


dataset_trainLabels = tf.data.Dataset.from_tensor_slices(trainLabels)
dataset_trainLabels = dataset_trainLabels.map(lambda trainLabels: tf.py_func(_read_py_function, [trainLabels], [tf.uint8]))
dataset_trainLabels = dataset_trainLabels.repeat(count=1000)
dataset_trainLabels = dataset_trainLabels.batch(batch_size)
iterator_trainLabels = dataset_trainLabels.make_one_shot_iterator()
next_element_trainLabels = iterator_trainLabels.get_next()
############################################VALIDATION##################################################
dataset_validationInputs = tf.data.Dataset.from_tensor_slices(validationInputs)
dataset_validationInputs = dataset_validationInputs.map(lambda validationInputs: tf.py_func(_read_py_function, [validationInputs], [tf.uint8]))
dataset_validationInputs= dataset_validationInputs.map(parse_function)
dataset_validationInputs= dataset_validationInputs.repeat(count=1000)
dataset_validationInputs= dataset_validationInputs.batch(batch_size)
iterator_validationInputs= dataset_validationInputs.make_one_shot_iterator()
next_element_validationInputs = iterator_validationInputs.get_next()

dataset_validationLabels= tf.data.Dataset.from_tensor_slices(validationLabels)
dataset_validationLabels = dataset_validationLabels.map(lambda validationLabels: tf.py_func(_read_py_function, [validationLabels], [tf.uint8]))
dataset_validationLabels = dataset_validationLabels.repeat(count=1000)
dataset_validationLabels= dataset_validationLabels.batch(batch_size)
iterator_validationLabels= dataset_validationLabels.make_one_shot_iterator()
next_element_validationLabels= iterator_validationLabels.get_next()


#define placeholders
xTraininginput=tf.placeholder(dtype = tf.float32, shape = [None,timeSteps,96],name = 'batch_x')
yTraininglabel=tf.placeholder(dtype = tf.int8, shape = [None,timeSteps,96],name = 'batch_y')
init_state = tf.placeholder(tf.float32, [number_of_layers, 2, batch_size, state_size])


#create cell state, hidden output pairs
state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(number_of_layers)])

#create LSTM stacked layers
def lstm_cell():
  return tf.contrib.rnn.LSTMCell(state_size, state_is_tuple = True)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(number_of_layers)])

#define dropout wrappers
stacked_lstm=tf.contrib.rnn.DropoutWrapper(stacked_lstm,input_keep_prob=0.2,output_keep_prob=1.0,state_keep_prob=0.5)

hidden_states, current_state = tf.nn.dynamic_rnn(stacked_lstm,inputs=xTraininginput,initial_state=rnn_tuple_state,time_major = False)
states_series_rnn_outputs = tf.reshape(hidden_states,[-1,state_size])
fc_layer1 = tf.contrib.layers.fully_connected(inputs = hidden_states, num_outputs = 96, activation_fn = tf.nn.relu)
fcc_softmax=tf.nn.softmax(fc_layer1)


#Loss layer
with tf.name_scope("losses"):

#Non-overlapping        
#    losses = tf.losses.softmax_cross_entropy(onehot_labels= yTraininglabel,logits=fc_layer1)
#Overlapping    
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=yTraininglabel,logits=fc_layer1)
    
#Adam Optimizer with exponential decay of learning rate
with tf.name_scope("optimizer"):
    global_step=tf.Variable(0, name='global_step', trainable=False)
    starter_learning_rate=0.01
    decay_steps=100
    
    learning_rate = tf.train.natural_exp_decay(starter_learning_rate, global_step,decay_steps, 0.95, staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss =losses,global_step=global_step)

    
with tf.name_scope("accuracy"):
    accuracy_op, accuracy= tf.metrics.accuracy(labels=tf.argmax(yTraininglabel) ,predictions= tf.argmax((fcc_softmax)))



tf.summary.scalar('losses', tf.reduce_mean(losses))
tf.summary.scalar('accuracy', tf.squeeze(accuracy))
merged_summary = tf.summary.merge_all()



with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    init = tf.global_variables_initializer()
    sess.run(init)
    train_writer = tf.summary.FileWriter(logs_path+'/train', graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(logs_path+'/test')

saver=tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    init = tf.global_variables_initializer()
    sess.run(init)
    
    #Loop for number of epochs
    for epoch in range(numberEpochs):
        
        loss_list=[]
        
        #Loop over the batches(training)
        for i in range(int(len(trainInputs)/batch_size)):
            
            _current_state = np.zeros((number_of_layers, 2, batch_size, state_size))
            inputImage = sess.run(next_element_trainInputs)
            midiValue = sess.run(next_element_trainLabels)
            
            inputImage=np.asarray(inputImage,dtype=np.float32)
            midiValue=np.asarray(midiValue,dtype=np.int8)
            

            midiValue=np.squeeze(midiValue,axis=0)
            
            
            feed_dict={
                    xTraininginput: inputImage,
                    yTraininglabel: midiValue,
                    init_state: _current_state
                }
            

            _optimizer, losses_, accuracy_ = sess.run([optimizer,losses,accuracy],feed_dict)
            
            loss_list.append(np.mean(losses_))
            summ,losses__,accu_train=sess.run([merged_summary,losses,accuracy],feed_dict=feed_dict)    
            train_writer.add_summary(summ,epoch)
        
        
        
        
        #Loop over the batches(validaton)
        for i in range(int(len(validationInputs)/batch_size)):
            _current_state = np.zeros((number_of_layers, 2, batch_size, state_size))
            inputImage = sess.run(next_element_validationInputs)
            midiValue = sess.run(next_element_validationLabels)
            
            inputImage=np.asarray(inputImage,dtype=np.float32)
            midiValue=np.asarray(midiValue,dtype=np.int8)
            
            midiValue=np.squeeze(midiValue,axis=0)
            
            feed_dict_vali={
                    xTraininginput: inputImage,
                    yTraininglabel: midiValue,
                    init_state: _current_state
                }
            vali_losses, vali_accuracy =sess.run([losses,accuracy],feed_dict_vali)
            summ,losses__,accu = sess.run([merged_summary,losses,accuracy], feed_dict=feed_dict_vali)
            test_writer.add_summary(summ,epoch)      



        
        Loss_Values=sum(loss_list)/len(loss_list)       
        print('Epoch :',epoch, ', Average_Loss:' , Loss_Values, ', Training_Loss:' , np.mean(losses_),  ', Validation_Loss:' , np.mean(vali_losses),', Train_Accu:' , accu_train, 'Validation_Accuracy:' , vali_accuracy)
 
        print('')
       
    


    saver=tf.train.Saver()
    saver.save(sess,logs_path)
    print('Model Saved')

train_writer.close()
test_writer.close()
