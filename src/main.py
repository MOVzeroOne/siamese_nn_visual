import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import pylab
import mpl_toolkits.mplot3d as Axis3D

transform = transforms.Compose([transforms.ToTensor()])


class datasubset():
    def __init__(self,amount_of_each=100,highest_number=10):
        self.train = torchvision.datasets.MNIST(".",train=True,transform=transform,download=True)
        self.highest_number = highest_number
        self.numbers_ordered = self.order_num(self.train)
        self.amount_of_each = amount_of_each
    
    def order_num(self,data):
        numbers_ordered = [[] for i in range(self.highest_number)]

        for num in tqdm(data,ascii=True,desc="prepare dataset"):
            if(len(numbers_ordered[num[1]]) <100):
                numbers_ordered[num[1]].append(num[0]) 

        return numbers_ordered

    def sample_num(self,number):
        image = self.numbers_ordered[number][np.random.randint(len(self.numbers_ordered[number]))]
        return image.view(1,1,28,28)
    
    def first_few(self,number,amount):
        return self.all_of_number(number)[:amount]

    def all_of_number(self,number):
        numbers = []
        for image in self.numbers_ordered[number]:
            numbers.append(image.view(1,1,28,28))
        return numbers
        


class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.Sequential(nn.Conv2d(1,3,3),nn.ReLU(),nn.Conv2d(3,6,3),nn.ReLU(),nn.Conv2d(6,1,3))
        self.layers = nn.Sequential(nn.Linear(484,242),nn.ReLU(),nn.Linear(242,3))

    def forward(self,x):
        return self.layers(self.convolution(x).view(x.size(0),-1))


def distance(x_0,x_1):
    return torch.sum(torch.abs(x_0-x_1),dim=1)

def contrastive_loss(x_0,x_1,y,m=1):
    #margin > 0
    #y = 1 is dissimilar and y = 0 is similair
    
    return torch.sum((1-y)*distance(x_0,x_1)*0.5 + (y)*torch.max(torch.zeros(x_0.size(0)),m-distance(x_0,x_1))*0.5)

def batch_sample(numbers_used,BATCH_SIZE,dataset):
    chosen = [np.random.choice(numbers_used,2) for _ in range(BATCH_SIZE)]
    batch_data = [[pair[0],pair[1],0] if pair[0] == pair[1] else [pair[0],pair[1],1] for pair in chosen]

    x_0,x_1,y = list(zip(*batch_data))
    x_0 = torch.cat([dataset.sample_num(i) for i in x_0])
    x_1 = torch.cat([dataset.sample_num(i) for i in x_1])
    
    y = torch.tensor(y,dtype=torch.float)
    return x_0,x_1,y

if __name__ == "__main__":
    dataset = datasubset()
    network = net()
    optimizer = optim.Adam(network.parameters(),lr=0.001)


    #HYPER PARAMETERS
    EPOCHS = 10000
    AMOUNT_PLOTTED_EACH = 10
    BATCH_SIZE = 10
    numbers_used = np.arange(10)
    PLOT_END_ONLY = False
    

    ### COLORS 
    NUM_COLORS = dataset.highest_number
    cm = pylab.get_cmap('gist_rainbow')
    cgen = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

    ### the rest

    x_plot = []
    y_plot = []
    plt.ion()
    fig = plt.figure(0)
    ax = fig.add_subplot(111,projection="3d")
    for i in tqdm(range(EPOCHS),ascii=True,desc="training"):
        optimizer.zero_grad()
        x_0_in,x_1_in,y = batch_sample(numbers_used,BATCH_SIZE,dataset)

        x_0 = network(x_0_in)
        x_1 = network(x_1_in)
 
        
        loss = contrastive_loss(x_0,x_1,y)
        loss.backward()
        optimizer.step()
        x_plot.append(i)
        y_plot.append(loss.detach().item())
        if(not PLOT_END_ONLY):
            for j in numbers_used:
                x_num_plot = []
                y_num_plot = []
                z_num_plot = []
                for image in dataset.first_few(j,AMOUNT_PLOTTED_EACH):
                    output = network(image)
                    
                    x_num_plot.append(output[0][0].detach().item())
                    y_num_plot.append(output[0][1].detach().item())
                    z_num_plot.append(output[0][2].detach().item())

                ax.scatter(x_num_plot,y_num_plot,z_num_plot,label=str(j),color=cgen[j])
                ax.legend()
                
            plt.pause(0.01)
            ax.cla()
            
        
    plt.ioff()
    for j in numbers_used:
        x_num_plot = []
        y_num_plot = []
        z_num_plot = []
        for image in dataset.first_few(j,AMOUNT_PLOTTED_EACH):
            output = network(image)
                
            x_num_plot.append(output[0][0].detach().item())
            y_num_plot.append(output[0][1].detach().item())
            z_num_plot.append(output[0][2].detach().item())

        ax.scatter(x_num_plot,y_num_plot,z_num_plot,label=str(j),color=cgen[j])
        ax.legend()
    plt.figure(1)
    plt.plot(x_plot,y_plot)
    plt.show()

