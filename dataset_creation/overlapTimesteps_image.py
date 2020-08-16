import pickle
import numpy as np
import os


overlapFactor=108
timeSteps=216
saveImagepickle=r'**Enter the path'

#pick 96xN pickle file
f = open(r'xx.pckl', 'rb')
imageObj = pickle.load(f)
f.close()


value=0
num=0

#Identical for both image and midi
length=int(((len(imageObj)+len(imageObj)/2)-216)/216)


for i in range(timeSteps):
    if i==0:
        image_data=imageObj[value,:]
        image_data_expandDims=np.expand_dims(image_data,axis=0)
        image_data_=image_data_expandDims
        value=value+1
    else:
        image_data=imageObj[value,:]
        image_data_expandDims=np.expand_dims(image_data,axis=0)
        image_data_ = np.concatenate((image_data_, image_data_expandDims), axis=0)
        value=value+1

num=num+1        
imageFile = ("imageFile%d" % (num))
f = open((os.path.join(saveImagepickle,imageFile + "." + 'pckl')), 'wb')
pickle.dump(image_data_,f)
f.close()
print(image_data_.shape)  
print(value)  

for i in range(length):
    value=value-108
    
    for j in range(timeSteps):
        if j==0:
            x=value
            image_data=imageObj[x,:]
            
            image_data_expandDims=np.expand_dims(image_data,axis=0)
            image_data_=image_data_expandDims
            value=value+1
        else:
           x=value
           image_data=imageObj[x,:]
           
           image_data_expandDims=np.expand_dims(image_data,axis=0)
           image_data_ = np.concatenate((image_data_, image_data_expandDims), axis=0)
           value=value+1
    
    num=num+1
    print(x)
    imageFile = ("imageFile%d" % (num))
    f = open((os.path.join(saveImagepickle,imageFile + "." + 'pckl')), 'wb')
    pickle.dump(image_data_,f)
    f.close()
    print(image_data_.shape)     

    
        




