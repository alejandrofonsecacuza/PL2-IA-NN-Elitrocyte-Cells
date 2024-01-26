#!/usr/bin/env python
# coding: utf-8

# In[26]:


import flask
from flask import Flask,request,render_template,jsonify
import torch
import os
from PIL import Image
import io
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_channels):
        super(Net,self).__init__()

        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride = 1, padding = 1)

        self.fc1 = nn.Linear(self.num_channels*4*10*10, self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 3)

    def forward(self,x):
        #Empieza 3x80x80
        x = self.conv1(x) # num_channels x 80 x 80
        x = F.relu(F.max_pool2d(x, 2)) # num_channels x 40 x 40
        x = self.conv2(x) # num_channels*2 x 40 x40
        x = F.relu(F.max_pool2d(x, 2)) #num_channels*2 x 20 x 20
        x = self.conv3(x) # num_channels*4 x20x20
        x = F.relu(F.max_pool2d(x, 2)) # num_channels*4 x 10 x 10

        #flatten
        x = x.view(-1, self.num_channels*4*10*10)

              #fc
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        #log_softmax

        x = F.log_softmax(x, dim=1)

        return x       







# In[27]:


path_base=os.getcwd()
model=Net(32)
model.load_state_dict(torch.load(path_base+'/Model_v2.0_ManyCells.pth'))
model.eval()


# In[28]:
print("Empezando..")

app=Flask(__name__)


# In[29]:


@app.route('/')
def index():
    return render_template('index.html')


# In[30]:


@app.route('/procesar_imagen',methods=['POST'])
def procesar_imagen():
    print("hola mundo")
     
    image_file=request.files['image']

    print('image_file')    
    img=Image.open(io.BytesIO(image_file.read()))
    trasnform=transforms.ToTensor()
    img_tensor=trasnform(img)
    print("Prediciendo....")
    #model.eval()
    outputs = model.forward(img_tensor)
    _, preds = torch.max(outputs,1)

    return jsonify({'predict':int(preds[0])})


# In[25]:


if __name__=='__main__':
    app.run(debug=True)


# In[ ]:




