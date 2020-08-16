'''
Forms 96x216 pickle file if the pickle format of dimension 96xN is supplied 
'''


import os
import pickle
import numpy as np


timeSteps=216
value=0
num=0
saveImagefile=r'**Enter the path**'



f = open(r'xx.pckl', 'rb')
imageObj = pickle.load(f)
f.close()





for j in range(int(len(imageObj)/timeSteps)):
    num=num+1
    image_data = np.array([])
    
    for i in range(216):
        
        image_data=imageObj[value,:]
        image_data_expandDims=np.expand_dims(image_data,axis=0)
        if i==0:
            image_data_=image_data_expandDims
            value=value+1
        else:    
            image_data_ = np.concatenate((image_data_, image_data_expandDims), axis=0)
            value=value+1
            
        if value % timeSteps==0:
            break
    
    print(value)
    imageFile = ("imageFile%d" % (num))
    f = open((os.path.join(saveImagefile,imageFile + "." + 'pckl')), 'wb')
    pickle.dump(image_data_,f)
    f.close()
