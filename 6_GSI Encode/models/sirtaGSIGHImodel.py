import torch
import torch.nn as nn
import torchvision.models as tvModels
from torchinfo import summary

# Define the ResNet-50 based model
class GSItoGHIresnet50(nn.Module):
    def __init__(self, num_features, gridSize, pretrained_weights_path=None):
        super(GSItoGHIresnet50, self).__init__()
        self.gridSize = gridSize
        self.resnet50 = tvModels.resnet50()
        if pretrained_weights_path:
            # Load pre-trained weights from the local file
            pretrained_weights = torch.load(pretrained_weights_path)
            self.resnet50.load_state_dict(pretrained_weights)
        # Modify the final fully connected layer to match the output size
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, 1024)  # 1024x1 vector
        self.reduceTo16 = nn.Sequential(
            # input is 1024 X 1
            nn.Linear(1024,512),
            nn.Tanh(),
            # state size. 512 X 1
            nn.Linear(512,128),
            nn.Tanh(),
            # state size. 128 X 1
            nn.Linear(128,self.gridSize*self.gridSize),
            nn.Tanh(),
            # state size. gridSize*gridSize X 1
        )
        self.estimateGHI = nn.Sequential(
            # input is self.gridSize*self.gridSize+num_features X 1
            nn.Linear((self.gridSize*self.gridSize)+num_features,8),
            nn.Tanh(),
            # state size. 8 X 1
            nn.Linear(8,4),
            nn.Tanh(),
            # state size. 4 X 1
            nn.Linear(4,1),
            # state size. 1 X 1
        )

    def forward(self, image, features):
        x = self.resnet50(image)
        x = self.reduceTo16(x)
        features = features.view(features.size(0), -1) # Flatten the features tensor to make it 2D
        # print("\tIn Model: x_16 size", x.size(),
        #       "features size", features.size())
        x = torch.concat([x, features], axis=1) # Concatenate with features
        x = self.estimateGHI(x)
        return x

'''///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////'''
# Define a custom acivation function 'sith' which is a product of sigmoid and tanh.
# It behaves similarly to 'silu' when x is -ve but similar to tanh when x is +ve.
# It also ensures that the function is continuous and differentiable everywhere.
def sith(x):
    return torch.sigmoid(x) * torch.tanh(x)

# Define a custom acivation function 'reth' which is a rectified version of tanh.
def reth(x):
    return torch.relu(torch.tanh(x))

# Define the model whichtakes SZA and SAA as input and identifies the sun position
# impact in the grid - defined by gridSize
class SZASAAtoGridPos(nn.Module):
    def __init__(self, gridSize):
        super(SZASAAtoGridPos, self).__init__()
        self.gridSize = gridSize # 4 for 4X4 grid
        self.solarPosParse = nn.Sequential(
            # input is 2 X 1
            nn.Linear(2,4),
            nn.Sigmoid(),# nn.Tanh(),
            # state size. 4 X 1
            nn.Linear(4,8),
            nn.Sigmoid(),# nn.Tanh(),
            # state size. 8 X 1
            nn.Linear(8,self.gridSize*self.gridSize),
            nn.Sigmoid(),# nn.Tanh(),
            # state size. self.gridSize*self.gridSize X 1
        )
    def forward(self, sza, saa):
        # print(torch.concat((sza, saa), axis=1).shape)
        output = self.solarPosParse(torch.concat((sza, saa), dim=1))
        return output

# Define the model which computes the final cloud impact score from feature arrays and
# SZA and SAA - using the SZASAAtoGridPos model for sunPos estimation
class CloudImpactNNnoSP(nn.Module):
    def __init__(self, gridSize):
        super(CloudImpactNNnoSP, self).__init__()
        self.gridSize = gridSize
        # Define parameter weights for each sky/cloud class - total classes = 5 {0,1,2,3,4}
        self.a = torch.nn.Parameter(torch.tensor(0.1))#(torch.randn(()))
        self.b = torch.nn.Parameter(torch.tensor(0.5))#(torch.randn(()))
        self.c = torch.nn.Parameter(torch.tensor(0.9))#(torch.randn(()))
        self.d = torch.nn.Parameter(torch.tensor(0.8))#(torch.randn(()))
        self.e = torch.nn.Parameter(torch.tensor(0.3))#(torch.randn(()))
        self.parseSZASAA = SZASAAtoGridPos(self.gridSize)

    def forward(self, cloudFraction, cloudClass, sza, saa):
        def modify_elements(tensor):
            modified_tensor = torch.where(tensor != 0, torch.zeros_like(tensor), torch.ones_like(tensor))
            return modified_tensor
        cloud_out = (self.a * torch.mul(cloudFraction, modify_elements(cloudClass)) +
                    self.b * torch.mul(cloudFraction, modify_elements(cloudClass - 1)) +
                    self.c * torch.mul(cloudFraction, modify_elements(cloudClass - 2)) +
                    self.d * torch.mul(cloudFraction, modify_elements(cloudClass - 3)) +
                    self.e * torch.mul(cloudFraction, modify_elements(cloudClass - 4)))
        cloud_out = reth(cloud_out)# torch.sigmoid(cloud_out)
        solar_pos_out = self.parseSZASAA(sza, saa)
        output = torch.mul(cloud_out, solar_pos_out)
        # output = cloud_out
        return output

    def string(self):
        return f'a = {self.a.item()} , b = {self.b.item()} , c = {self.c.item()} , d = {self.d.item()} , e = {self.e.item()}'


# Define the model which computes the final cloud impact score from feature arrays and
# estimated sunPos array
class CloudImpactNNwithSP(nn.Module):
    def __init__(self):
        super(CloudImpactNNwithSP, self).__init__()
        # Define parameter weights for each sky/cloud class - total classes = 5 {0,1,2,3,4}
        self.a = torch.nn.Parameter(torch.tensor(0.1))#(torch.randn(()))
        self.b = torch.nn.Parameter(torch.tensor(0.5))#(torch.randn(()))
        self.c = torch.nn.Parameter(torch.tensor(0.9))#(torch.randn(()))
        self.d = torch.nn.Parameter(torch.tensor(0.8))#(torch.randn(()))
        self.e = torch.nn.Parameter(torch.tensor(0.3))#(torch.randn(()))
        # Define parameter weights for each sun-position class - total classes = 5 {0,1,2,3,4}
        self.v = torch.nn.Parameter(torch.randn(()))
        self.w = torch.nn.Parameter(torch.randn(()))
        self.x = torch.nn.Parameter(torch.randn(()))
        self.y = torch.nn.Parameter(torch.randn(()))
        self.z = torch.nn.Parameter(torch.randn(()))

    def forward(self, cloudFraction, cloudClass, sun_pos):
        def modify_elements(tensor):
            modified_tensor = torch.where(tensor != 0, torch.zeros_like(tensor), torch.ones_like(tensor))
            return modified_tensor
        cloud_out = (self.a * torch.mul(cloudFraction, modify_elements(cloudClass)) +
                    self.b * torch.mul(cloudFraction, modify_elements(cloudClass - 1)) +
                    self.c * torch.mul(cloudFraction, modify_elements(cloudClass - 2)) +
                    self.d * torch.mul(cloudFraction, modify_elements(cloudClass - 3)) +
                    self.e * torch.mul(cloudFraction, modify_elements(cloudClass - 4)))
        cloud_out = reth(cloud_out)
        cloudSP_out = (self.v * modify_elements(sun_pos) +
                       self.w * modify_elements(sun_pos - 1) +
                       self.x * modify_elements(sun_pos - 2) +
                       self.y * modify_elements(sun_pos - 3) +
                       self.z * modify_elements(sun_pos - 4))
        cloudSP_out = torch.tanh(cloudSP_out)
        output = torch.mul(cloud_out, cloudSP_out)
        return output

    def string(self):
        return f'y = {self.a.item()} , {self.b.item()} , {self.c.item()} , {self.d.item()} , {self.e.item()}, {self.v.item()}, {self.w.item()} , {self.x.item()} , {self.y.item()} , {self.z.item()}'

'''///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////'''
# Define the main model to estimate GHI from the cloud features, sun_pos (or SZA, SAA),
# and CSM models
class CloudImpactNNtoGHI(nn.Module):
    def __init__(self, gridSize, spFlag=False):
        super(CloudImpactNNtoGHI, self).__init__()
        self.gridSize = gridSize
        self.spFlag = spFlag
        if spFlag:
            self.getCloudImpactVectorwithSP = CloudImpactNNwithSP()
        else:
            self.getCloudImpactVectornoSP = CloudImpactNNnoSP(self.gridSize)
        self.impactVectorAndCSMToGHI = nn.Sequential(
            # input is (self.gridSize*self.gridSize + 1) X 1
            nn.Linear((self.gridSize*self.gridSize)+1,8), 
            nn.Tanh(),
            # state size. 8 X 1
            nn.Linear(8,4),
            nn.Tanh(),
            # state size. 4 X 1
            nn.Linear(4,1),
            nn.Tanh(),
            # state size. 1 X 1
        )
        self.impactVectorToImpactScalar = nn.Sequential(
            # input is self.gridSize*self.gridSize X 1
            nn.Linear(self.gridSize*self.gridSize,8), 
            nn.Tanh(),
            # state size. 8 X 1
            nn.Linear(8,4),
            nn.Tanh(),
            # state size. 4 X 1
            nn.Linear(4,1),
            nn.Tanh(),
            # state size. 1 X 1
        )
        self.impactCSMToGHI = nn.Sequential(
            # input is 2 X 1
            nn.Linear(2,4),
            nn.Tanh(),
            # state size. 4 X 1
            nn.Linear(4,1),
            # state size. 1 X 1
        )

    def forward(self, cloudFraction, cloudClass, csm, sza=None, saa=None, sun_pos=None):
        if self.spFlag and sun_pos is not None:
            impactVector = self.getCloudImpactVectorwithSP(cloudFraction, cloudClass, sun_pos)
        elif sza is not None and saa is not None:
            impactVector = self.getCloudImpactVectornoSP(cloudFraction, cloudClass, sza, saa)
        else:
            raise Exception('Invalid attribute error in forward pass!')
        impactScalar = self.impactVectorToImpactScalar(impactVector)
        output = self.impactCSMToGHI(torch.cat((impactScalar,csm), dim=1))
        '''
        output = self.impactVectorAndCSMToGHI(torch.cat((impactVector,csm), dim=1))
        '''
        # output = impactScalar
        return output

    def getSZASAAoutput(self, sza, saa):
        if (not self.spFlag) and (sza is not None) and (saa is not None):
            return self.getCloudImpactVectornoSP.parseSZASAA(sza, saa)
        else:
            return None
    
    def getImpactVector(self, cloudFraction, cloudClass, csm, sza=None, saa=None, sun_pos=None):
        """
        Return the cloud impact score vector for a given datapoint
        """
        if self.spFlag and sun_pos is not None:
            out = self.getCloudImpactVectorwithSP(cloudFraction, cloudClass, sun_pos)
            out_str = self.getCloudImpactVectorwithSP.string()
            return out, out_str
        elif sza is not None and saa is not None:
            out = self.getCloudImpactVectornoSP(cloudFraction, cloudClass, sza, saa)
            out_str = self.getCloudImpactVectornoSP.string()
            return out, out_str
        else:
            return None

    def getImpactScalar(self, cloudFraction, cloudClass, csm, sza=None, saa=None, sun_pos=None):
        """
        Return the cloud impact scalar score for a given datapoint
        """
        if self.spFlag and sun_pos is not None:
            impactVector = self.getCloudImpactVectorwithSP(cloudFraction, cloudClass, sun_pos)
        elif sza is not None and saa is not None:
            impactVector =  self.getCloudImpactVectornoSP(cloudFraction, cloudClass, sza, saa)
        else:
            return None
        return self.impactVectorToImpactScalar(impactVector)

    def string(self):
        if self.spFlag:
            return self.getCloudImpactVectorwithSP.string()
        else:
            return self.getCloudImpactVectornoSP.string()


'''////////////////////////////////////////////////////////////////////////////////'''
if __name__=="__main__":
    # Instantiate the model
    model = GSItoGHIresnet50(num_features=3, gridSize=4, pretrained_weights_path='./savedWeights/resnet50-11ad3fa6.pth')  # 3 features: SZA, SAA, CSM

    batch_size = 256
    summary(model, [(batch_size, 3, 256, 256), (batch_size, 3)])

    gridSize = 4
    net = CloudImpactNNtoGHI(gridSize, spFlag = False)
    batch_size = 32
    summary(net, [(batch_size,gridSize*gridSize), (batch_size,gridSize*gridSize), (batch_size,1), (batch_size,1), (batch_size,1)])#, (batch_size,gridSize*gridSize)])