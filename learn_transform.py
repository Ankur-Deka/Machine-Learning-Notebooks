import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class simpleNet(torch.nn.Module):
    def __init__(self, inDim, outDim):
        super(simpleNet, self).__init__()
        self.fc1 = nn.Linear(inDim, 10)
        self.fc2 = nn.Linear(10, outDim)

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

    funcArr = ['angToVec', 'vecToAng']
    inDimArr = [1,2]
    outDimArr = [2,1]
    ang = 2*math.pi*torch.rand((1000,1)).view(-1,1)             # 0-2pi
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    vec = torch.cat((cos, sin), dim = 1)

    angVal = torch.arange(0,2*math.pi,0.01).view(-1,1)
    cosVal = torch.cos(angVal)
    sinVal = torch.sin(angVal)
    vecVal = torch.cat((cosVal, sinVal), dim = 1)

    inArr = [ang, vec]
    outArr = [vec, ang]
    inValArr = [angVal, vecVal]
    outValArr = [vecVal, angVal]

    for activation in ['relu', 'sigmoid']:
        for i,func in enumerate(funcArr):
            if i == 0:
                continue
            inDim, outDim = inDimArr[i], outDimArr[i]
            myNet = simpleNet(inDim, outDim)
            optimizer = torch.optim.Adam(myNet.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            x, y = inArr[i], outArr[i]
            xVal, yVal = inValArr[i], outValArr[i]
      
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
                    pred = myNet(xVal, activation)
                    if inDim == 1:
                        for j in range(outDim):
                            plt.subplot(2,1,j+1)
                            plt.scatter(xVal.detach().numpy(),yVal[:,j].detach().numpy(),label = 'true')
                            plt.scatter(xVal.detach().numpy(),pred[:,j].detach().numpy(),label = 'learnt')
                            plt.legend(loc = 'best')
                    elif outDim == 1:
                        for j in range(inDim):
                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(xVal[:,0].detach().numpy(), xVal[:,1].detach().numpy(), yVal.detach().numpy(), label = 'true')
                            ax.scatter(xVal[:,0].detach().numpy(), xVal[:,1].detach().numpy(), pred.detach().numpy(), label = 'learnt')
                            plt.legend(loc = 'best')
                        #     plt.scatter(yVal[:,j].detach().numpy(),myNet(xVal[:,j], activation).detach().numpy(), label = 'learnt')
                    # plt.subplot(2,1,2)
                    saveLoc = os.path.join(saveDir, str(int(e*10/epochs)))
                    plt.savefig(saveLoc, dpi = 400)
                    plt.show()
                    plt.close()
                loss.backward()
                optimizer.step()