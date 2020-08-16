'''
Form 96xN pickle file if the path spectrogram and midi files are given
'''



'''
Make pickle for Image data

'''

import numpy as np
import pickle
import natsort
import os
import cv2

pickImagepath=r'**Enter path'
imageFiles=[]
for d in os.listdir(pickImagepath):
    imageFiles.append(os.path.join(pickImagepath, d))

imageFiles = natsort.natsorted(imageFiles,reverse=False)
input_data = np.array([])

for i in range(len(imageFiles)):
    img = cv2.imread(imageFiles[i],1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.transpose(gray)
    if i==0:
        input_data=gray
    else:    
        input_data = np.concatenate((input_data, gray), axis=0)
        
        

f = open(r'xx.pckl', 'wb')
pickle.dump(input_data, f)
f.close()
print(input_data.shape)
####################################################################################################################################
'''
Make pickle for Midi data

'''
pickMidipath=r'**Enter path'
midiFiles=[]
for d in os.listdir(pickMidipath):
    midiFiles.append(os.path.join(pickMidipath, d))

midiFiles = natsort.natsorted(midiFiles,reverse=False)
input_data = np.array([])

for i in range(len(midiFiles)):
    img = cv2.imread(midiFiles[i],1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.transpose(gray)
    if i==0:
        input_data=gray
    else:    
        input_data = np.concatenate((input_data, gray), axis=0)
        
    
f = open(r'yy.pckl', 'wb')
pickle.dump(input_data, f)
f.close()
print(input_data.shape)

