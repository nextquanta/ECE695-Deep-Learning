# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:40:05 2020

@author: risij
"""

#from google.colab import drive
#drive.mount('/content/drive')
#%cd drive/My\ Drive/ECE695


'''
RESNET Experiment with Imagenet data:

While doing hw with Imagenet although I wrote whole code by myself but object oriented structure of code is inspired from
Prof Kak , DLstudio block. With imagenet , I couldn't go beyond 25 layers because Colab is getting crashed and it's taking
longer time to train.


RESNET Experiment with CIFAR data:

With regards , While doing experiment with CIFAR data, I took "run_code_for training" and "run_code_for testing" method from DL studio
And picked class BMEnet, Skipblock from DL studio and did changes in these two classes. Following are accuracies with CIFAR data

All accuracies mentioned are for one epoch:

First Try: Instead of just adding identity , adding convolution of identity which gave accuracy of 49%
Second Try: Adding one more fully connected layer with 2000 units and as expected accuracy improved 
to 61%.
Third Try: Add one more fully connected layer with conv in skip path didn't see any significant improvement.
Fourth Try: Grouped convolution in skip blocks with 2 filter groups and got an accuracy of 53%.
Fifth try: Adding post activation layer after skip connection addition and got an accuracy of 51% 
Sixth try: Increase skip blocks from 16 to 32 and got an accuracy of 55%
'''


f=open("output.txt","w")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt        
import numpy as np
import pdb



dtype=torch.float

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DL():
    def __init__(self,epochs):
        self.epochs=epochs
        
    def load_imagenet_dataset(self,train_path,test_path):       
        transform = tvt.Compose([tvt.Resize((128,128)),tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_path="imagenet_train"
        test_path="imagenet_test"
        train_dataset = torchvision.datasets.ImageFolder(root=train_path,transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=4,num_workers=0,shuffle=True)
        test_dataset = torchvision.datasets.ImageFolder(root=test_path,transform=transform)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=4,num_workers=0,shuffle=True)
    
    
    
    def run_code_for_training(self,net):
        net = net.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-6, momentum=0.9)
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if(i%50==49):
                  average_loss= running_loss/  float(50)
                  print("\n[epoch:%d, batch:%5d] loss: %.3f" %(epoch + 1, i + 1, average_loss))
                  running_loss = 0.0
            f.write("Epoch %d: %.3f\n" % (epoch + 1, average_loss))
      
    def run_code_for_testing(self,net):
      net = net.to(device)
      inc=0
      loop_num=0
      for i, data in enumerate(self.test_loader):
          inputs, labels = data
          inputs = inputs.to(device)
          labels = labels.to(device)
          outputs = net(inputs) 
          for j in range(len(labels)):
            loop_num+=1
            if(labels[j]==torch.argmax(outputs[j])):
              inc+=1
      f.write("Classification Accuracy: %d %%" % (100 * inc / float(loop_num)))






class Skip_class(nn.Module):
    def __init__(self,in_ch,out_ch,pool_enable=False):
        super(Skip_class, self).__init__()
        self.in_ch=in_ch
        self.out_ch=out_ch
        self.pool_enable=pool_enable
        self.skip_convo1 = nn.Conv2d(in_ch//2, out_ch//2, 3,padding=1)
        self.skip_convo2 = nn.Conv2d(out_ch//2, out_ch//2, 3,padding=1)
        if(pool_enable):
            self.pool = nn.MaxPool2d(2, 2)  
        self.batch_norm = nn.BatchNorm2d(out_ch//2)        
        
    def forward(self,x):
        identity1=x[:,:self.in_ch//2,:,:]  
        out1= x[:,:self.in_ch//2,:,:]
        out1=F.relu(self.batch_norm(self.skip_convo1(out1)))
        out1=self.batch_norm(self.skip_convo2(out1))

        if(self.pool_enable):
          identity1=self.pool(identity1)
          out1=self.pool(out1)

        for i in range(int(self.out_ch/self.in_ch)):
          out1[:,(self.in_ch//2)*i:(self.in_ch//2)*(i+1),:,:] += identity1    


        identity2=x[:,self.in_ch//2:,:,:]  
        out2= x[:,self.in_ch//2:,:,:]
        out2=F.relu(self.batch_norm(self.skip_convo1(out2)))
        out2=self.batch_norm(self.skip_convo2(out2))

        if(self.pool_enable):
          identity2=self.pool(identity2)
          out2=self.pool(out2)

        for i in range(int(self.out_ch/self.in_ch)):
          out2[:,(self.in_ch//2)*i:(self.in_ch//2)*(i+1),:,:] += identity2

        out = torch.cat([out1, out2], dim=1)               

        return out



class Main_class(nn.Module):
    def __init__(self):
        super(Main_class, self).__init__()
        self.resolution_down_count=4
        self.conv1 = nn.Conv2d(3, 256, 3,padding=1) 
        self.conv2 = nn.Conv2d(256, 256, 3,padding=1) 
        self.pool = nn.MaxPool2d(2, 2)     

        self.skip256= Skip_class(256,256)
        self.skip256_512=Skip_class(256,512,True)
        self.skip512=Skip_class(512,512)



        self.fc1 = nn.Linear((512*128*128)//(2**(2*self.resolution_down_count)), 1000) 
        self.fc2 = nn.Linear(1000, 5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
  
        for i in range(4):
          x=self.skip256(x)
        
        x=self.skip256_512(x)
 
        for i in range(4):       
          x=self.skip512(x)

        x=self.pool(x)

        x = x.view(-1, (512*128*128)//(2**(2*self.resolution_down_count)))
        x=self.fc1(x)
        x=self.fc2(x)
        return x

dl= DL(5)

model=Main_class()
dl.load_imagenet_dataset("imagenet_train","imagenet_test")
dl.run_code_for_training(model)
dl.run_code_for_testing(model)


f.close()
