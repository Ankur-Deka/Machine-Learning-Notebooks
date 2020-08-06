import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import os

class simpleNet(torch.nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x, activation):
        if activation == 'relu':
            x = torch.relu(self.fc1(x))
        elif activation == 'sigmoid':
            x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    learning_rate = 1e-2
    epochs = 10000

    myNet = simpleNet()

    optimizer = torch.optim.Adam(myNet.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    x1 = 2*math.pi*torch.rand((1000,1))             # 0-2pi
    x2 = 2*(torch.rand((1000,1))-0.5)               # -1 to 1
    x3range = 50                    
    x3 = 2*x3range*(torch.rand((1000,1))-0.5)       # -50 to 50
    
    xArr = [x1, x1, x1, x2, x2, x3]
    yArr = [torch.sin(x1), torch.cos(x1), torch.tan(x1), torch.asin(x2), torch.acos(x2), torch.atan(x3)]
    funcArr = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']

    x1Val = torch.arange(0,2*math.pi,0.01).view(-1,1)
    x2Val = torch.arange(-1,1,0.01).view(-1,1)
    x3Val = torch.arange(-x3range, x3range, 0.01).view(-1,1)
    
    xValArr = [x1Val, x1Val, x1Val, x2Val, x2Val, x3Val]
    yValArr = [torch.sin(x1Val), torch.cos(x1Val), torch.tan(x1Val), torch.asin(x2Val), torch.acos(x2Val), torch.atan(x3Val)]
    
    for activation in ['relu', 'sigmoid']:
        for i,x in enumerate(xArr[:1]):
            func = funcArr[i]
            y = yArr[i]
            xVal, yVal = xValArr[i], yValArr[i]
            saveDir = os.path.join('figures',activation,func)
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            
            print('----------------------------------------')
            print('Activation: {}, Function to learn: {}'.format(activation, func))
            print('----------------------------------------') 
            
            for e in range(epochs):
                optimizer.zero_grad()
                pred = myNet(x, activation)
                loss = criterion(y, pred)
                if not e%(epochs/10):
                    print('Epoch: {}, loss: {}'.format(e,loss))
                    plt.scatter(xVal.detach().numpy(),yVal.detach().numpy(),label = 'true')
                    plt.scatter(xVal.detach().numpy(),myNet(xVal, activation).detach().numpy(), label = 'learnt')
                    plt.legend(loc = 'best')
                    saveLoc = os.path.join(saveDir, str(int(e*10/epochs)))
                    plt.savefig(saveLoc, dpi = 400)
                    plt.close()
                loss.backward()
                optimizer.step()