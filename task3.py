# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:50:53 2020

@author: risij
"""

'''
------------------------- Code References ----------------------------------------------
In this code class "PurdueShapes5Dataset_individual_Dataset" and "PurdueShapes5Dataset"
structure is inspired from DL studio class build by Prof Avinash Kak. I have done modification
specially in constructor part for merging images for all noise labels and making it little
bit more generic.
----------------------------------------------------------------------------------------


-------------------------Architecture Description------------------------------------------



In this network I am using first few convo and skip layers then fully connected layer fc1
from this layer , I have created two path one is for noise classification and one is 
for shape classification.
In noise classifcation I am using two more fully connected layers for noise classification
then softmax layer as an output layer.
In shape classification path I first I am reshaping fc1 output into matrix of 128x4 and then
multiplying with one hot vector of noise classification path. The idea I have is to let it flow
only part of information from previous layers depending upon noise outputs.After multiplication
I am using two more fully conncted layer in shape classification path.

For backprogation I am using one loss for boundary box regression and one for total classification loss
which is sum of both noise classification loss and shape classification loss.
------------------------------------------------------------------------------------------------

-----------------------  Training-Testing and Result -------------------------------------
For training I have merged all datasets of 40,000 images and testing individually on 10,000 
images for each noise label.
For boundary box L2 norm error for coordinates was 2.6 which resemble roughly one 1-2 pixel 
dfference.
For noise classification accuracy: 99.025%
For 0% noise shape classification accuracy: 89.3%
For 20% noise shape classification accuracy: 81.8%
For 50% noise shape classification accuracy: 71.9%
For 80% noise shape classification accuracy: 58.3%

FOr 80% noise training accuracy is coming good but in testing accuracy is not good.
So it might be due to complex model.
I am submitting hw with current implementation but will do experiments to see if
it improves by introducing simpler model.
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
from numpy import pi
import pdb



torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dtype=torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PurdueShapes5Dataset_individual_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, dataset_file, transform=None,noise_filter=False):
        super(PurdueShapes5Dataset_individual_Dataset, self).__init__()
        self.noise_filter=noise_filter
        pt_file="torch-saved-"+dataset_file.split(".gz")[0]+".pt"
        map_pt_file="torch-saved-PurdueShapes5-label-map.pt"
        if(not(os.path.exists(pt_file) and os.path.exists(map_pt_file))):
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


class PurdueShapes5Dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, dataset_file, test_train,transform=None,noise_filter=False):
        super(PurdueShapes5Dataset, self).__init__()
        self.noise_filter=noise_filter
        self.test_train=test_train
        if(self.test_train=="train"):
          pt_file="torch-saved-PurdueShapes5.pt"
          map_pt_file="torch-saved-PurdueShapes5-label-map.pt"
          all_data = {}
          noise_label_dict= {"0":0,"20":1,"50":2,"80":3}
          if(not(os.path.exists(pt_file) and os.path.exists(map_pt_file))):
            start_index=0
            for k in range(len(dataset_file)):
              noise_label=0
              if("noise" in dataset_file[k]):
                noise_label= noise_label_dict[dataset_file[k].split(".gz")[0].split("-")[-1]]
              f = gzip.open(os.path.join(dataroot,dataset_file[k]), 'rb')
              dataset = f.read()
              if sys.version_info[0] == 3:
                  self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
              else:
                  self.dataset, self.label_map = pickle.loads(dataset)
              data_len=len(self.dataset)
              
              [all_data.update({ list(self.dataset.keys())[i]+start_index:list(self.dataset.values())[i]+[noise_label] }) for i in range(data_len)]
              start_index=start_index+data_len
            self.all_data=all_data
            torch.save(self.all_data, pt_file)
            torch.save(self.label_map, map_pt_file)
        else:
          pt_file="torch-saved-PurdueShapes5_test.pt"
          map_pt_file="torch-saved-PurdueShapes5-label-map_test.pt"
          all_data = {}
          noise_label_dict= {"0":0,"20":1,"50":2,"80":3}
          if(not(os.path.exists(pt_file) and os.path.exists(map_pt_file))):
            start_index=0
            for k in range(len(dataset_file)):
              noise_label=0
              if("noise" in dataset_file[k]):
                noise_label= noise_label_dict[dataset_file[k].split(".gz")[0].split("-")[-1]]
              f = gzip.open(os.path.join(dataroot,dataset_file[k]), 'rb')
              dataset = f.read()
              if sys.version_info[0] == 3:
                  self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
              else:
                  self.dataset, self.label_map = pickle.loads(dataset)
              data_len=len(self.dataset)
              
              [all_data.update({ list(self.dataset.keys())[i]+start_index:list(self.dataset.values())[i]+[noise_label] }) for i in range(data_len)]
              start_index=start_index+data_len
            self.all_data=all_data
            torch.save(self.all_data, pt_file)
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
        im_tensor = torch.zeros(3,32,32, dtype=torch.float)
        im_tensor[0,:,:] = torch.from_numpy(R)
        im_tensor[1,:,:] = torch.from_numpy(G)
        im_tensor[2,:,:] = torch.from_numpy(B)
        bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
        if(self.noise_filter):
            im_tensor = ndimage.gaussian_filter(im_tensor, 2)
        sample = {'image' : im_tensor, 
                  'bbox' : bb_tensor,
                  'label' : self.dataset[idx][4],
                  'noise_label': self.dataset[idx][5]}
        if self.transform:
             sample = self.transform(sample)
        return sample




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
    def __init__(self,skip_block_depth):
        super(Main_class, self).__init__()
        #self.num_classification_output_units=num_classification_output_units
        #self.num_external_input_units=num_external_input_units
        #self.external_input=external_input
        self.skip_block_depth = skip_block_depth
        self.resolution_down_count_classification=1
        self.resolution_down_count_regression=1

        
        #######################################################################################################
        self.conv1 = nn.Conv2d(3, 64, 3,padding=1) 
        self.conv2 = nn.Conv2d(64, 64, 3,padding=1) 
        self.conv3 = nn.Conv2d(64, 64, 3,padding=1) 
        self.pool = nn.MaxPool2d(2, 2)     
        
        self.skip64_list = nn.ModuleList([Skip_class(64,64) for _ in range(self.skip_block_depth)])
        self.skip64_128=Skip_class(64,128,False)
        self.skip128_list=nn.ModuleList([Skip_class(128,128) for _ in range(self.skip_block_depth)])


        self.noise_in_dim_unit=128
        self.noise_label_len = 4
        self.noise_in_dim = self.noise_label_len*self.noise_in_dim_unit
        self.fc1 = nn.Linear((128*32*32)//(2**(2*self.resolution_down_count_classification)), self.noise_in_dim) 

        self.fc2 = nn.Linear(self.noise_in_dim, self.noise_in_dim*2)
        self.fc3 = nn.Linear(self.noise_in_dim*2, self.noise_label_len)                  
        ########################################################################################


        ############################ BBOX ##############################

        self.fc4 = nn.Linear((64*32*32)//(2**(2*self.resolution_down_count_regression)), 1000) 
        self.fc5 = nn.Linear(1000,200) 
        self.fc6 = nn.Linear(200, 4)

        ##################################################################################
        self.fc7 = nn.Linear(int(self.noise_in_dim/self.noise_label_len), 200)
        self.fc8 = nn.Linear(200, 5)
        ################################################################


        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        bb_x = x.clone()  
    
        x = F.relu(self.conv2(x))
        for i in range(self.skip_block_depth):
          x=self.skip64_list[i](x)
        
        x=self.skip64_128(x)
 
        for i in range(self.skip_block_depth):       
          x=self.skip128_list[i](x)

        
        x = x.view(-1, (128*32*32)//(2**(2*self.resolution_down_count_classification))) 

        x=self.fc1(x)                                          ##This will be resshaped
        x_noise_in = x.clone()
        x_noise_in = self.fc2(x_noise_in)
        x_noise_in = self.fc3(x_noise_in)     
        
        x_noise_in = torch.softmax(x_noise_in,dim=1)            ## output of noise classifer




        ######################################
        
        bb_x = bb_x.view(-1, (64*32*32)//(2**(2*self.resolution_down_count_regression)))
        bb_x=self.fc4(bb_x)
        bb_x=F.relu(self.fc5(bb_x))
        bb_x= self.fc6(bb_x)

        ###########################################
        x = x.view((-1,self.noise_in_dim_unit,self.noise_label_len))
        #x = torch.matmul(x,x_noise_in.view(-1,self.noise_label_len,1)).view((-1,self.noise_in_dim_unit))
        x = torch.matmul(x,torch.zeros(x_noise_in.shape).to(device).scatter(1,torch.argmax(x_noise_in,1).unsqueeze(1),1.0).view(-1,self.noise_label_len,1)).view((-1,self.noise_in_dim_unit))

        
        x = self.fc7(x)
        x = self.fc8(x)
        ##########################################
        return x,bb_x,x_noise_in



class DL():
    def __init__(self,epochs):
        self.epochs=epochs
        
    def load_purdueshape_dataset(self,train_dataset,test_dataset):       
        self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=8,num_workers=0,shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,num_workers=0,shuffle=True)

    def load_purdueshape_dataset_testonly(self,test_dataset):       
        self.test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,num_workers=0,shuffle=True)   
    

    def run_code_for_training(self,net):
        net = net.to(device)
        criterion1 = torch.nn.CrossEntropyLoss()                     ##shape classification
        criterion2 = nn.MSELoss()                                    ##Boundary box regression
        criterion3 = torch.nn.CrossEntropyLoss()                     ##noise classification 
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
        for epoch in range(self.epochs):
            running_total_loss = 0.0
            running_loss_regression = 0.0 
            for i, data in enumerate(self.train_loader):
                inputs = [data['image'],data['bbox']]
                labels = data['label']
                noise_labels = data['noise_label']
                inputs[0] = inputs[0].to(device)
                inputs[1] = inputs[1].to(device)
                labels = labels.to(device)
                noise_labels = noise_labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs[0])
                loss_classification = criterion1(outputs[0], labels)   
                loss_noise_classification = criterion3(outputs[2], noise_labels)
                total_loss = loss_classification+loss_noise_classification
                total_loss.backward(retain_graph=True) 

                loss_regression = criterion2(outputs[1], inputs[1])
                loss_regression.backward(retain_graph=True)   

                optimizer.step()
                running_total_loss += total_loss.item()
                running_loss_regression += loss_regression.item()
                
                print_depth=len(self.train_loader)-1
                print_depth=1000
                if(i%print_depth==((print_depth-1))):
                  average_total_loss= running_total_loss/  float(print_depth)
                  average_loss_regression= running_loss_regression/  float(print_depth)

                  print("\n[Epoch:%d, batch:%5d] Total loss: %.3f, Regrssion loss: %.3f" %(epoch + 1, i + 1, average_total_loss,average_loss_regression))
                  running_total_loss = 0.0
                  running_loss_regression = 0.0
            torch.save(net.state_dict(), "epoch_"+str(epoch)+'_weights_only.pth')
      
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

    def run_code_for_testing_noiselabel(self,net,print_data=""):
      net = net.to(device)
      classification_inc=0
      loop_num=0
      regression_error=0
      CM = torch.zeros((4,4))
      for i, data in enumerate(self.test_loader):
          inputs = [data['image'],data['bbox']]
          labels = data['noise_label']
          inputs[0] = inputs[0].to(device)
          inputs[1] = inputs[1].to(device)
          labels = labels.to(device)
          outputs = net(inputs[0])
          for j in range(len(labels)):
            loop_num+=1
            if(labels[j]==torch.argmax(outputs[2][j])):
              classification_inc+=1
            CM[labels[j]][torch.argmax(outputs[2][j])]+=1


      print("Noise Classification Accuracy: ",100 * classification_inc / float(loop_num))
      torch.set_printoptions(sci_mode=False)
      print_data = print_data+"Noise Classification Accuracy: {0} %\n".format((100 * classification_inc / float(loop_num)))
      print_data = print_data+"Noise Confusion Matrix:\n{0}\n".format(CM)
      return print_data



weight_file_path="epoch_15_weights_only.pth"
num_epochs=25
dataroot=r"/content/drive/My Drive/ECE695/HW05/data"
train_file_name_list=["PurdueShapes5-10000-train.gz","PurdueShapes5-10000-train-noise-20.gz","PurdueShapes5-10000-train-noise-50.gz","PurdueShapes5-10000-train-noise-80.gz"]
test_file_name_list=["PurdueShapes5-1000-test.gz","PurdueShapes5-1000-test-noise-20.gz","PurdueShapes5-1000-test-noise-50.gz","PurdueShapes5-1000-test-noise-20.gz"]
p_train = PurdueShapes5Dataset(dataroot,train_file_name_list,test_train="train",noise_filter=False)
p_test = PurdueShapes5Dataset(dataroot,test_file_name_list,test_train="test",noise_filter=False)

dl= DL(num_epochs)

model=Main_class(3)
dl.load_purdueshape_dataset(p_train,p_test)
dl.run_code_for_training(model)

print_data = ""

test_file_name_list=["PurdueShapes5-1000-test.gz","PurdueShapes5-1000-test-noise-20.gz","PurdueShapes5-1000-test-noise-50.gz","PurdueShapes5-1000-test-noise-20.gz"]
p_test = PurdueShapes5Dataset(dataroot,test_file_name_list,test_train="test",noise_filter=False)
dl.load_purdueshape_dataset_testonly(p_test)
model.load_state_dict(torch.load(weight_file_path))
print_data = dl.run_code_for_testing_noiselabel(model,print_data)


test_file_name="PurdueShapes5-1000-test.gz"
p_test = PurdueShapes5Dataset_individual_Dataset(dataroot,test_file_name)
dl.load_purdueshape_dataset_testonly(p_test)
model.load_state_dict(torch.load(weight_file_path))
print_data = dl.run_code_for_testing(model,print_data)
print_data=print_data.replace("noiselabel","0")


test_file_name="PurdueShapes5-1000-test-noise-20.gz"
p_test = PurdueShapes5Dataset_individual_Dataset(dataroot,test_file_name)
dl.load_purdueshape_dataset_testonly(p_test)
model.load_state_dict(torch.load(weight_file_path))
print_data = dl.run_code_for_testing(model,print_data)
print_data=print_data.replace("noiselabel","20")


test_file_name="PurdueShapes5-1000-test-noise-50.gz"
p_test = PurdueShapes5Dataset_individual_Dataset(dataroot,test_file_name)
dl.load_purdueshape_dataset_testonly(p_test)
model.load_state_dict(torch.load(weight_file_path))
print_data = dl.run_code_for_testing(model,print_data)
print_data=print_data.replace("noiselabel","50")


test_file_name="PurdueShapes5-1000-test-noise-80.gz"
p_test = PurdueShapes5Dataset_individual_Dataset(dataroot,test_file_name)
dl.load_purdueshape_dataset_testonly(p_test)
model.load_state_dict(torch.load(weight_file_path))
print_data = dl.run_code_for_testing(model,print_data)
print_data=print_data.replace("noiselabel","80")



print(print_data)
f.write(print_data)

f.close()


