import tensorflow as tf
import os

tf.reset_default_graph()

#Enter the path where log files needs to be saved
logs_path = r"**Enter the path**"


#Enter the paths where images reside
labelsPath=r"**Enter the path**"
inputsPath=r"**Enter the path**"

#Declare hyperparameters
num_epochs = 1
batch_size = 1


def createfilepaths(labelsPath,inputsPath):

    labelsList=[]
    imagesList=[]
    for d in os.listdir(labelsPath):
        labelsList.append(os.path.join(labelsPath, d))

    for d in os.listdir(inputsPath):
        imagesList.append(os.path.join(inputsPath, d))    
    
    return labelsList,imagesList
    
labelsList,imagesList = createfilepaths(labelsPath,inputsPath)

trainLabels = labelsList[:int(len(labelsList)*0.8)]
trainInputs = imagesList[:int(len(imagesList)*0.8)]
    
validationLabels = labelsList[-int(len(labelsList)*0.2):]
validationInputs = imagesList[-int(len(imagesList)*0.2):]


def parse_function(labels, images):
    
    label_string = tf.read_file(labels)
    image_string = tf.read_file(images)
    
    label_decode = tf.image.decode_png(label_string)
    image_decode = tf.image.decode_png(image_string)
    
    label_converted = tf.image.convert_image_dtype(label_decode,dtype=tf.float32)
    image_converted = tf.image.convert_image_dtype(image_decode,dtype=tf.float32)

    return label_converted, image_converted


#datapipeline for training
dataset = tf.data.Dataset.from_tensor_slices((trainLabels, trainInputs))
dataset = dataset.repeat(count=500)
dataset = dataset.map(parse_function, num_parallel_calls=1)
dataset = dataset.batch(batch_size)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


#datapipeline for validation
dataset_validation = tf.data.Dataset.from_tensor_slices((validationLabels,validationInputs))
dataset_validation = dataset_validation.repeat(count=500)
dataset_validation = dataset_validation.map(parse_function)
dataset_validation = dataset_validation.batch(batch_size)
iterator_validation = dataset_validation.make_one_shot_iterator()
next_element_validation = iterator_validation.get_next()
    

#define placeholders
xTraininglabel = tf.placeholder(tf.float32,shape= [None,96, 216,1])
yTraininginput = tf.placeholder(tf.float32, shape= [None,96,216,1])


#define layers(convolutional layer)
def new_conv_layer(inputs, num_outputs, kernel_size, stride, name):
    with tf.variable_scope(name):
       output =  tf.contrib.layers.conv2d(inputs=inputs, num_outputs=num_outputs,kernel_size=kernel_size,stride = stride,data_format = "NHWC",  activation_fn=tf.nn.relu, scope = name)
       return output  

#define layers(deconvolutional layer)
def new_layer(inputs, num_outputs, kernel_size, stride, name):
    with tf.variable_scope(name):
        layer = tf.contrib.layers.conv2d_transpose(inputs=inputs, num_outputs=num_outputs,kernel_size=kernel_size,stride = stride,data_format = "NHWC", activation_fn=tf.nn.relu, scope = name)
        return layer

#define layers(relu)
def new_relu_layer(input, name):
    with tf.variable_scope(name):
        layer = tf.nn.relu(input,name=name)
        return layer    
    
#connect the layers
layer_conv1 = new_conv_layer(inputs=yTraininginput, num_outputs=32, kernel_size=[5,5], stride=1, name ="conv1")
layer_conv2 = new_conv_layer(inputs=layer_conv1, num_outputs=64, kernel_size=[5,5], stride=1, name ="conv2")
layer_conv3 = new_conv_layer(inputs=layer_conv2, num_outputs=64, kernel_size=[5,5], stride=1, name ="conv3")
layer_conv4 = new_conv_layer(inputs=layer_conv3, num_outputs=128, kernel_size=[5,5], stride=1, name ="conv4")
layer_deconv1 = new_layer(inputs=layer_conv4, num_outputs = 1,  kernel_size=[5,5],stride=1 ,name = 'deconv1')





#MSE Loss
with tf.name_scope("loss"):
    mse = tf.losses.mean_squared_error(labels = xTraininglabel,predictions = layer_deconv1, scope='loss')
    
#Adam Optimizer    
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(loss = mse)
    
#PSNR
with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.image.psnr(a=layer_deconv1,b=xTraininglabel,max_val=1.0, name='accuracy'))
    
    
tf.summary.scalar('loss', tf.squeeze(mse))
tf.summary.scalar('accuracy', tf.squeeze(accuracy))
merged_summary = tf.summary.merge_all()
saver = tf.train.Saver()

#Session 
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter(logs_path+'/train', graph=tf.get_default_graph())
    writer1 = tf.summary.FileWriter(logs_path+'/test', graph=tf.get_default_graph())      

    
    
    
    #Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
    writer1.add_graph(sess.graph)
    #Loop for number of epochs
    for epoch in range(num_epochs):
        
        
        loss_list=[]
        #Loop over the batches(training)
        for batch in range(int(len(trainLabels)/batch_size)):

                        
            x_label,y_input = sess.run(next_element)
            
            feed_dict_train = {yTraininginput : y_input,xTraininglabel : x_label}

            optimizer_train,mse_train,accuracy_train=sess.run([optimizer,mse,accuracy], feed_dict=feed_dict_train)   

            loss_list.append(mse_train)
            
        #Loop over the batches(validaton)
        for j in range(int(len(validationLabels)/batch_size)):   
            
            x_labelval, y_inputval = sess.run(next_element_validation)        
            feed_dict_vali = {yTraininginput : y_inputval, xTraininglabel : x_labelval} 
            mse_vali,accuracy_vali=sess.run([mse,accuracy], feed_dict=feed_dict_train)          
            
            
        
            
        
        Loss_Values=sum(loss_list)/len(loss_list)       
        print('Epoch :',epoch, ', Average_Loss:' , Loss_Values, ', Training_Loss:' , mse_train,  ', Validation_Loss:' , mse_vali, 'Validation_Accuracy:', accuracy_vali,'Validation_Accuracy:',accuracy_train)
        print('')
        
        summ_train = sess.run(merged_summary, feed_dict=feed_dict_train)
        summ_vali =  sess.run(merged_summary, feed_dict = feed_dict_vali)
        
        writer.add_summary(summ_train, epoch)
        writer1.add_summary(summ_vali, epoch)
        
        
    
    saver.save(sess, './cnn_model')
    print('Model saved')
  
writer.close()
writer1.close()
