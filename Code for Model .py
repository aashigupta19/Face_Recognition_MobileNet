#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications import MobileNet

#mobile net works on 224*224 images
row_img = 224
col_img = 224

#remove the FC layers
#extract model without top layer/ FC layers

MobileNet = MobileNet(weights = 'imagenet', include_top = False,
                      input_shape = (row_img, col_img, 3))

#now freeze the critical layers
#layers.trainable  = False

for layer in MobileNet.layers:
    layer.trainable = False
    
#to see the name of layers
for (i, layer) in enumerate(MobileNet.layers):
    print(str(i) + "  " + layer.__class__.__name__, layer.trainable)
    
    


# In[ ]:





# In[2]:


def fc_head(bottom_model, num_classes):
    #creates the top or head of model
    #placed onto top of layers next to it
    
    #shift tab won't work here 
    #coz Dense has really not been imported yet xD
    #creating our FC layers here
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation = 'relu')(top_model)
    top_model = Dense(1024, activation = 'relu')(top_model)
    top_model = Dense(512,  activation = 'relu')(top_model)
    top_model = Dense(num_classes, 
                            activation = 'softmax')(top_model)
    
    return top_model

#function end here

    


# In[ ]:





# In[3]:


#now that we've created our FC,
# add it to the next layers (the critical layers we left untouched)

# import all required libraries here

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import GlobalAveragePooling2D, Conv2D
from keras.layers import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

# Batch Normalization
# It is used to normalize the input layer by re-centering and re-scaling
# gives better performance as computing reduces
# gives stability to ANN


# In[4]:


#set our class number
#which is - the number of different labels of faces we're gonna predict from
#eg - here i have 3: Mom dad me

num_classes = 3                                          

Head_fc = fc_head(MobileNet, num_classes)

#The `Model` class adds training & evaluation routines to a `Network`.
model = Model(inputs = MobileNet.input, outputs = Head_fc)


# In[5]:


model.summary()


# In[ ]:





# In[ ]:





# In[6]:


#load our faces database into code

from keras.preprocessing.image import ImageDataGenerator

train_directory_path = 'C://Users//Aashi Gupta//Downloads//AI ML Ops 2020//Face Recognition using DL//Faces//train'
test_directory_path  = 'C://Users//Aashi Gupta//Downloads//AI ML Ops 2020//Face Recognition using DL//Faces//test'


# In[7]:


#now we have to process the images a little
#so as to fit our model perfectly
#also give better accuracy :)

#c/a Data Augmentation

train_datagen = ImageDataGenerator( rescale = 1./255,
                                    rotation_range= 45,
                                    width_shift_range= 0.3,
                                    height_shift_range= 0.3,
                                    horizontal_flip= True,
                                    fill_mode= 'nearest'
                                  )

test_datagen =  ImageDataGenerator( rescale= 1./255)


# In[ ]:





# In[ ]:





# In[8]:


#set our batch_size
batch_size = 32


# In[9]:


#how to load dataset from train directory onto variable                           

train_generator = train_datagen.flow_from_directory( train_directory_path,
                                                     target_size= (row_img, col_img),
                                                     batch_size= batch_size,
                                                     class_mode= 'categorical'
                                                   )


# In[ ]:





# In[10]:


#how to load dataset from test directory onto variable                             

test_generator = test_datagen.flow_from_directory(   test_directory_path,
                                                     target_size= (row_img, col_img),
                                                     batch_size= batch_size,
                                                     class_mode= 'categorical'
                                                   )


# In[ ]:





# In[ ]:





# In[ ]:


#now add functionality to our model


# In[11]:


from keras.optimizers import RMSprop
from keras.callbacks  import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("face_recog_mobilenet.h5",
                              monitor= "val_loss",
                              mode= "min",
                              save_best_only= True,
                              verbose = 1
                            )



earlystop = EarlyStopping( monitor= "val_loss",
                           min_delta= 0,
                           patience= 3,
                           verbose = 1,
                           restore_best_weights= True
                         )


# In[12]:


#we put our callbacks in a callback list
callbacks = [earlystop, checkpoint]


# In[13]:


# use a small learning rate for better convergence
#categorical classification it is
model.compile( loss = 'categorical_crossentropy',
               optimizer = RMSprop(learning_rate=0.001),
               metrics = ['accuracy']
             )


# In[28]:


72*3


# In[ ]:





# In[ ]:





# In[15]:


#number of training samples and test samples                               

train_samples = 216
test_samples  = 84


#number of epochs we want our model to run
epochs = 5
batch_size = 16


# In[16]:


history = model.fit_generator( train_generator,
                               steps_per_epoch= train_samples // batch_size,
                               epochs= epochs,
                               callbacks= callbacks,
                               validation_data= test_generator,
                               validation_steps= test_samples // batch_size
                             )


# In[17]:


from keras.models import load_model

face_recognition_model = load_model("face_recog_mobilenet.h5")


# In[ ]:





# In[18]:


#display of result when test data is fed

import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


# In[29]:


face_recog_dict =   { "[0]" : "Aashi",
                      "[1]" : "Mom",
                      "[2]" : "Dad"
                    }


face_recog_dict_n = { "n0"  : "Aashi",
                      "n1"  : "Mom",
                      "n2"  : "Dad"
                    }


# In[ ]:





# In[20]:


def draw_test(name, pred, im):
    face = face_recog_dict[str(pred)]
    BLACK = [0, 0, 0]   #defining RGB values for black color
    
    #copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]]) -> d
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100, cv2.BORDER_CONSTANT, value = BLACK)
    
    # putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
    # Draws a text string
    cv2.putText(expanded_image, face, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)


# In[21]:


#this function loads a random image from a random folder in our testing path
def getRandomImage(path):
    
    #Built-in mutable sequence.
    #If no argument is given, the constructor creates a new empty list.
    folders = list( filter( lambda x : os.path.isdir( os.path.join(path, x)),  os.listdir(path)))
    
    random_directory = np.random.randint(0, len(folders))
    path_class       = folders[random_directory]
    
    print("Class - " + face_recog_dict_n[str(path_class)])
    file_path = path + path_class
    
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0, len(file_names))
    image_name = file_names[random_file_index]
    
    return cv2.imread(file_path + "//" + image_name)
    


# In[27]:


for i in range(0,3):
    input_im = getRandomImage( "C://Users//Aashi Gupta//Downloads//AI ML Ops 2020//Face Recognition using DL//Faces//test//" )
    
    input_original = input_im.copy()
   
    #resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst
    #Resizes an image.
    input_original = cv2.resize( input_original, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR )
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im/255.
    input_im = input_im.reshape(1, 224, 224, 3)
    
   
    #getting prediction
    #np.argmax(a, axis=None, out=None)
    
    #Returns the indices of the maximum values along an axis.
    res = np.argmax( face_recognition_model.predict( input_im, 1, verbose = 0), axis = 1)
    
    
    #show image with predicted class
    draw_test("Prediction", res, input_original)
    cv2.waitKey(0)
    

cv2.destroyAllWindows()    
#end of code :))))) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




