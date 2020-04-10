# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:29:23 2020

@author: risij
"""

'''
------------------------- Code References ----------------------------------------------
In this code class  "PurdueShapes5Dataset" structure is inspired from DL studio class build 
by Prof Avinash Kak.I have done little bit modification to make it more generic.
Besides this Skipblock and Loadnet classes are copied exactly from DLstudio class build 
by Prof Avinash Kak.
----------------------------------------------------------------------------------------

-------------------------- Architecture/Filter Description ------------------------------
Loadnet2 Archictecture as mentioned in problem statement have been used. For smoothning images
I tried median filters and gaussian filter. With Gaussian filter and speifically with 
standard deviation of 4 I got best accuracy.
------------------------------------------------------------------------------------------

-----------------------  Training-Testing and Result -------------------------------------
Result are after 5 epochs
For 0% noise shape classification accuracy: 80.2%
For 20% noise shape classification accuracy: 77.3%
For 50% noise shape classification accuracy: 81.9%
For 80% noise shape classification accuracy: 75.4%
-------------------------------------------------------------------------------------------

'''


f=open("output.txt","w")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import gzip
import pickle
import sys
from scipy import ndimage

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



dtype=torch.float

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PurdueShapes5Dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, dataset_file, transform=None,noise_filter=False):
        super(PurdueShapes5Dataset, self).__init__()
        self.noise_filter=noise_filter
        pt_file="torch-saved-"+dataset_file.split(".gz")[0]+".pt"
        map_pt_file="torch-saved-PurdueShapes5-label-map.pt"
        if(not(os.path.exists(pt_file) and os.path.exists(map_pt_file))):
          print("\nLoading training data from the torch-saved archive")
          print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                  """the dataset for this script. First time loading could take\n"""
                  """a minute or so.  Any subsequent attempts will only take\n"""
                  """a few seconds.\n\n\n""")
          f = gzip.open(os.path.join(dataroot,dataset_file), 'rb')
          dataset = f.read()
          if sys.version_info[0] == 3:
              self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
          else:
              self.dataset, self.label_map = pickle.loads(dataset)
          torch.save(self.dataset, pt_file)
          torch.save(self.label_map, map_pt_file)
          
        self.dataset = torch.load(pt_file)
        self.label_map = torch.load(map_pt_file)

        self.class_labels = dict(map(reversed, self.label_map.items()))
        self.transform = transform           
     
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        r = np.array( self.dataset[idx][0] )
        g = np.array( self.dataset[idx][1] )
        b = np.array( self.dataset[idx][2] )
        R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
        if(self.noise_filter):
          R = ndimage.gaussian_filter(R, 4)
          G = ndimage.gaussian_filter(G, 4)
          B = ndimage.gaussian_filter(B, 4)

        im_tensor = torch.zeros(3,32,32, dtype=torch.float)
        im_tensor[0,:,:] = torch.from_numpy(R)
        im_tensor[1,:,:] = torch.from_numpy(G)
        im_tensor[2,:,:] = torch.from_numpy(B)
        bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
        if(self.noise_filter):
            im_tensor = ndimage.gaussian_filter(im_tensor, 2)
        
        sample = {'image' : im_tensor, 
                  'bbox' : bb_tensor,
                  'label' : self.dataset[idx][4] }
        if self.transform:
             sample = self.transform(sample)
        return sample

class SkipBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
      super(SkipBlock, self).__init__()
      self.downsample = downsample
      self.skip_connections = skip_connections
      self.in_ch = in_ch
      self.out_ch = out_ch
      self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
      norm_layer = nn.BatchNorm2d
      self.bn = norm_layer(out_ch)
      if downsample:
          self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

    def forward(self, x):
      identity = x                                     
      out = self.convo(x)                              
      out = self.bn(out)                              
      out = torch.nn.functional.relu(out)
      if self.in_ch == self.out_ch:
          out = self.convo(out)                              
          out = self.bn(out)                              
          out = torch.nn.functional.relu(out)
      if self.downsample:
          out = self.downsampler(out)
          identity = self.downsampler(identity)
      if self.skip_connections:
          if self.in_ch == self.out_ch:
              out += identity                              
          else:
              out[:,:self.in_ch,:,:] += identity
              out[:,self.in_ch:,:,:] += identity
      return out


class LOADnet2(nn.Module):
    """
    The acronym 'LOAD' stands for 'LOcalization And Detection'.
    LOADnet2 uses both convo and linear layers for regression
    """ 
    def __init__(self, skip_connections=True, depth=32):
      super(LOADnet2, self).__init__()
      self.pool_count = 3
      self.depth = depth // 2
      self.conv = nn.Conv2d(3, 64, 3, padding=1)
      self.pool = nn.MaxPool2d(2, 2)
      self.skip64 = SkipBlock(64, 64, 
                                                  skip_connections=skip_connections)
      self.skip64ds = SkipBlock(64, 64, 
                                  downsample=True, skip_connections=skip_connections)
      self.skip64to128 = SkipBlock(64, 128, 
                                                  skip_connections=skip_connections )
      self.skip128 = SkipBlock(128, 128, 
                                                    skip_connections=skip_connections)
      self.skip128ds = SkipBlock(128,128,
                                  downsample=True, skip_connections=skip_connections)
      self.fc1 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
      self.fc2 =  nn.Linear(1000, 5)
      ##  for regression
      self.conv_seqn = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
          nn.ReLU(inplace=True)
      )
      self.fc_seqn = nn.Sequential(
          nn.Linear(16384, 1024),
          nn.ReLU(inplace=True),
          nn.Linear(1024, 512),
          nn.ReLU(inplace=True),
          nn.Linear(512, 4)
      )

    def forward(self, x):
      x = self.pool(torch.nn.functional.relu(self.conv(x)))          
      ## The labeling section:
      x1 = x.clone()
      for _ in range(self.depth // 4):
          x1 = self.skip64(x1)                                               
      x1 = self.skip64ds(x1)
      for _ in range(self.depth // 4):
          x1 = self.skip64(x1)                                               
      x1 = self.skip64to128(x1)
      for _ in range(self.depth // 4):
          x1 = self.skip128(x1)                                               
      x1 = self.skip128ds(x1)                                               
      for _ in range(self.depth // 4):
          x1 = self.skip128(x1)                                               
      x1 = x1.view(-1, 128 * (32 // 2**self.pool_count)**2 )
      x1 = torch.nn.functional.relu(self.fc1(x1))
      x1 = self.fc2(x1)
      ## The Bounding Box regression:
      x2 = self.conv_seqn(x)
      x2 = self.conv_seqn(x2)
      # flatten
      x2 = x2.view(x.size(0), -1)
      x2 = self.fc_seqn(x2)
      return x1,x2


class DL():
    def __init__(self,epochs):
        self.epochs=epochs
        
    def load_purdueshape_dataset(self,train_dataset,test_dataset):       
        self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=4,num_workers=0,shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,num_workers=0,shuffle=True)
    
    def load_purdueshape_dataset_testonly(self,test_dataset):       
        self.test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,num_workers=0,shuffle=True)       
    
    def run_code_for_training(self,net,weight_file_name):
        net = net.to(device)
        criterion1 = torch.nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
        for epoch in range(self.epochs):
            running_loss_classification = 0.0
            running_loss_regression = 0.0 
            for i, data in enumerate(self.train_loader):
                inputs = [data['image'],data['bbox']]
                labels = data['label']
                inputs[0] = inputs[0].to(device)
                inputs[1] = inputs[1].to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = net(inputs[0])

                loss_classification = criterion1(outputs[0], labels)
                loss_classification.backward(retain_graph=True)    
                loss_regression = criterion2(outputs[1], inputs[1])
                loss_regression.backward()  
                
                optimizer.step()
                running_loss_classification += loss_classification.item()
                running_loss_regression += loss_regression.item()
                
                print_depth=500
                if(i%print_depth==print_depth-1):
                  average_loss_classification= running_loss_classification/  float(print_depth)
                  average_loss_regression= running_loss_regression/  float(print_depth)
                  print("\n[epoch:%d, batch:%5d] classification loss: %.3f, Regrssion loss: %.3f " %(epoch + 1, i + 1, average_loss_classification,average_loss_regression))
                  running_loss_classification = 0.0
                  running_loss_regression = 0.0
        torch.save(net.state_dict(), weight_file_name)    
      
    def run_code_for_testing(self,net,print_data=""):
      net = net.to(device)
      classification_inc=0
      loop_num=0
      regression_error=0
      CM = torch.zeros((5,5))
      for i, data in enumerate(self.test_loader):
          inputs = [data['image'],data['bbox']]
          labels = data['label']
          inputs[0] = inputs[0].to(device)
          inputs[1] = inputs[1].to(device)
          labels = labels.to(device)
          outputs = net(inputs[0])
          for j in range(len(labels)):
            loop_num+=1
            if(labels[j]==torch.argmax(outputs[0][j])):
              classification_inc+=1
            CM[labels[j]][torch.argmax(outputs[0][j])]+=1

          if(i%(len(self.test_loader)/10) == 0):
            print(data['bbox'],outputs[1].data[0].cpu())
          regression_error=regression_error+torch.dist(inputs[1],outputs[1])
      regression_error=regression_error/len(self.test_loader)

      print("Classification Accuracy: ",100 * classification_inc / float(loop_num))
      print("Regrssion_error L2 norm: ",regression_error)
      torch.set_printoptions(sci_mode=False)
      print_data = print_data+"Datasetnoiselabel Classification Accuracy: {0} %\n".format((100 * classification_inc / float(loop_num)))
      print_data = print_data+"Datasetnoiselabel Confusion Matrix:\n{0}\n".format(CM)
      return print_data


print_data = ""
num_epoch=5
dl= DL(num_epoch)
model=LOADnet2(4)

dataroot=r"/content/drive/My Drive/ECE695/HW05/data"
print_data=""

weight_file_name='weights_only_task2_noise_0.pth'
train_file_name="PurdueShapes5-10000-train.gz"
test_file_name="PurdueShapes5-1000-test.gz"

p_train = PurdueShapes5Dataset(dataroot,train_file_name,noise_filter=True)
p_test = PurdueShapes5Dataset(dataroot,test_file_name,noise_filter=True)

dl.load_purdueshape_dataset(p_train,p_test)
dl.run_code_for_training(model,weight_file_name)


dl.load_purdueshape_dataset_testonly(p_test)
model.load_state_dict(torch.load(weight_file_name))
print_data = dl.run_code_for_testing(model,print_data)
print_data=print_data.replace("noiselabel","0")


weight_file_name='weights_only_task2_noise_20.pth'
train_file_name="PurdueShapes5-10000-train-noise-20.gz"
test_file_name="PurdueShapes5-1000-test-noise-20.gz"

p_train = PurdueShapes5Dataset(dataroot,train_file_name,noise_filter=True)
p_test = PurdueShapes5Dataset(dataroot,test_file_name,noise_filter=True)

dl.load_purdueshape_dataset(p_train,p_test)
dl.run_code_for_training(model,weight_file_name)


dl.load_purdueshape_dataset_testonly(p_test)
model.load_state_dict(torch.load(weight_file_name))
print_data = dl.run_code_for_testing(model,print_data)
print_data=print_data.replace("noiselabel","20")


weight_file_name='weights_only_task2_noise_50.pth'
train_file_name="PurdueShapes5-10000-train-noise-50.gz"
test_file_name="PurdueShapes5-1000-test-noise-50.gz"

p_train = PurdueShapes5Dataset(dataroot,train_file_name,noise_filter=True)
p_test = PurdueShapes5Dataset(dataroot,test_file_name,noise_filter=True)

dl.load_purdueshape_dataset(p_train,p_test)
dl.run_code_for_training(model,weight_file_name)


dl.load_purdueshape_dataset_testonly(p_test)
model.load_state_dict(torch.load(weight_file_name))
print_data = dl.run_code_for_testing(model,print_data)
print_data=print_data.replace("noiselabel","50")


weight_file_name='weights_only_task2_noise_80.pth'
train_file_name="PurdueShapes5-10000-train-noise-80.gz"
test_file_name="PurdueShapes5-1000-test-noise-80.gz"

p_train = PurdueShapes5Dataset(dataroot,train_file_name,noise_filter=True)
p_test = PurdueShapes5Dataset(dataroot,test_file_name,noise_filter=True)

dl.load_purdueshape_dataset(p_train,p_test)
dl.run_code_for_training(model,weight_file_name)


dl.load_purdueshape_dataset_testonly(p_test)
model.load_state_dict(torch.load(weight_file_name))
print_data = dl.run_code_for_testing(model,print_data)
print_data=print_data.replace("noiselabel","80")

f.write(print_data)
f.close()
