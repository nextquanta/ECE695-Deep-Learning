import torchvision.transforms as tvt
import torchvision
import torch
from torch.utils.data import Sampler
import pdb
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np


transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


dataroot = "./data/CIFAR10"
## Define where the training and the test datasets are located:
train_data_loc = torchvision.datasets.CIFAR10(root=dataroot, train=True,download=True, transform=transform)
test_data_loc = torchvision.datasets.CIFAR10(root=dataroot, train=False,download=True,transform=transform)
## Now create the data loaders:

train_data_loc = [item for item in train_data_loc if(item[1]==3 or item[1]==5)]
test_data_loc = [item for item in test_data_loc if(item[1]==3 or item[1]==5)]


trainloader = torch.utils.data.DataLoader(train_data_loc, batch_size=5,shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_data_loc, batch_size=5, shuffle=False, num_workers=0)


probability_class = {3:[0,1],5:[1,0]}

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N, D_in, H1, H2, D_out = 8, 3*32*32, 1000, 256, 2
# Randomly initialize weights
w1 = torch.randn(D_in, H1, device=device, dtype=dtype)
w2 = torch.randn(H1, H2, device=device, dtype=dtype)
w3 = torch.randn(H2, D_out, device=device, dtype=dtype)
learning_rate = 1e-9
#for t in range(500):
#min_loss=very high number

epoch_num=200
min_loss = float('inf')
loss_list=[]
file_data=""
for t in range(epoch_num):
    loss_sum=0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        x = inputs.view(inputs.size()[0], -1)
        h1 = x.mm(w1) ## In numpy, you would say h1 = x.dot(w1)
        h1_relu = h1.clamp(min=0)
        h2 = h1_relu.mm(w2)
        h2_relu = h2.clamp(min=0)
        y_pred = h2_relu.mm(w3)
        y= torch.FloatTensor([probability_class[int(item)] for item in labels]).to(device)
        # Compute and print loss
        #pdb.set_trace()
        loss = (y_pred - y).pow(2).sum().item()
        #if (t % 10 == 0 and (i==0)):
        #    print(t, loss)
        
        loss_sum=loss_sum+loss
        
        #backprop to compute gradients of loss wrt to parameters w1,w2 and w3
        # Backprop to compute gradients of w1 and w2 with respect to loss <=== WRONG WRONG WRONG
        y_error = y_pred - y
        grad_w3 = h2_relu.t().mm(2 * y_error) #<<<<<< Gradient of Loss w.r.t w3
        h2_error =  y_error.mm(w3.t()) # backpropagated error to the h2 hidden layer
        grad_h2_error=h2.clone()
        grad_h2_error[h2 < 0] = 0 # We set those elements of the backpropagated error
        grad_h2_error[h2 > 0] = 1 
        grad_w2 = h1_relu.t().mm(grad_h2_error) #<<<<<< Gradient of Loss w.r.t w2
        
        h1_error =  h2_error.mm(w2.t()) # backpropagated error to the h1 hidden layer
        grad_h1_error=h1.clone()
        grad_h1_error[h1 < 0] = 0 # We set those elements of the backpropagated error
        grad_h1_error[h1 > 0] = 1
        grad_w1 = x.t().mm(grad_h1_error) #<<<<<< Gradient of Loss w.r.t w2
        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        w3 -= learning_rate * grad_w3
        
        #print(loss)
        #pdb.set_trace()
    file_data=file_data+"Epoch {} : {}\n".format(t,loss_sum/(5*len(trainloader)))
    

    loss_list.append(loss_sum/(5*len(trainloader)))
    if(min_loss>loss_sum):
      min_loss=loss_sum
      W1=w1
      W2=w2
      W3=w3
        
        
plt.plot(range(epoch_num),loss_list)        
plt.show()


correct_classification=0
for i, data in enumerate(testloader): 
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    x = inputs.view(inputs.size()[0], -1)
    h1 = x.mm(W1) ## In numpy, you would say h1 = x.dot(w1)
    h1_relu = h1.clamp(min=0)
    h2 = h1_relu.mm(W2)
    h2_relu = h2.clamp(min=0)
    y_pred = h2_relu.mm(W3)
    y= torch.FloatTensor([probability_class[int(item)] for item in labels]).to(device)
    
    
    #y_pred = F.softmax(y_pred, dim = 1)
    #pdb.set_trace()
    for i in range(5):
      if(y_pred[i][0]>y_pred[i][1]):
        y_pred[i][0]=1;
        y_pred[i][1]=0;
      else:
        y_pred[i][0]=0;
        y_pred[i][1]=1;
        
    correct_classification = correct_classification+np.sum(np.all(y_pred.cpu().numpy()==y.cpu().numpy(),axis=1))
    
    #print(y)

#print(correct_classification)  
#print(correct_classification*100/(5*len(testloader)))
file_data=file_data+"Test Accuracy : {}%".format(correct_classification*100/(5*len(testloader)))
f=open("output.txt","w")
f.write(file_data)
f.close()
