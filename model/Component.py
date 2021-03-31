import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms
import numpy as np

def Tensor2Image(img):
    img = img.cpu()
    img = img * 0.5 + 0.5
    img = transforms.ToPILImage()(img)
    return img

def weights_init_normal(m):
    if isinstance(m, nn.ConvTranspose2d):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Conv2d):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        
def one_hot(label, depth):
    # label 转 onehot
    out_tensor = torch.zeros(len(label), depth)
    for i, index in enumerate(label):
        out_tensor[i][index] = 1.0
    return out_tensor

def one_hot_long(label, depth, set_intensity=None):
    # N 倍长度的onehot    [-1~ 1]
    out_tensor = torch.zeros(len(label), depth*5)
    
    for i, index in enumerate(label):
        if set_intensity:
            onehot = torch.Tensor(5).uniform_(0.1*set_intensity, 0.1*set_intensity)
        else:
            onehot = torch.Tensor(5).uniform_(-1, 1)
        out_tensor[i] = (-1*torch.abs(onehot)).repeat(depth)
        out_tensor[i][index*5: (index+1)*5] = torch.abs(onehot)
    return out_tensor

def one_hot_intensity(label, depth):
    # 强度0.1-1之间的onehot
    out_tensor = torch.zeros(len(label), depth)
    intensity = [0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8, 0.9, 1.0]
    for i, index in enumerate(label):
        out_tensor[i][index] = intensity[np.random.randint(9)]
    return out_tensor

def concat_label(feature_map, label, duplicate=1):
    feature_shape = feature_map.shape
    if duplicate<1:
        return feature_map
    label = label.repeat(1, duplicate)
    label_shape = label.shape
    if len(feature_shape) == 2:
        return torch.cat((feature_map, label), 1)
    elif len(feature_shape) == 4:
        label = label.view(feature_shape[0], label_shape[-1], 1, 1)
        return torch.cat((feature_map, label*torch.ones((feature_shape[0], label_shape[-1], feature_shape[2], feature_shape[3])).cuda()), 1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        conv = [nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),# 3*128*128 --> 64*64*64
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),# 64*64*64 --> 128*32*32
                 nn.BatchNorm2d(128),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),# 128*32*32 --> 256*16*16
                 nn.BatchNorm2d(256),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),# 256*16*16 --> 512*8*8
                 nn.BatchNorm2d(512),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2),# 512*8*8 --> 1024*4*4
                 nn.BatchNorm2d(1024),
                 nn.ReLU(inplace=True)
                ]
        self.conv = nn.Sequential(*conv)
        
        fc = [nn.Linear(1024*4*4, 50),
                 nn.Tanh()
                  ]
        self.fc = nn.Sequential(*fc)
    
    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 1024*4*4)
        x_z = self.fc(x)
        
        return x_z

class Decoder(nn.Module):
    def __init__(self, N_expr=6, N_long=5, N_z=50):
        super(Decoder, self).__init__()
        self.duplicate = int(N_z//N_expr)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(N_z+N_long*N_expr*self.duplicate, 1024*4*4)
        # (1024+30)*4*4 --> 512*8*8
        dconv1 = [nn.Upsample(scale_factor=2),
                 nn.Conv2d(1024+N_long*N_expr, 512, kernel_size=3, stride=1, padding=1, bias=False),
                 nn.BatchNorm2d(512),
                 nn.ReLU(inplace=True)
                ]
        self.dconv1 = nn.Sequential(*dconv1) 
        
        dconv2 = [nn.Upsample(scale_factor=2),
                 nn.Conv2d(512+N_long*N_expr, 256, kernel_size=3, stride=1, padding=1, bias=False),
                 nn.BatchNorm2d(256),
                 nn.ReLU(inplace=True)
                ]
        self.dconv2 = nn.Sequential(*dconv2) 
        
        dconv3 = [nn.Upsample(scale_factor=2),
                 nn.Conv2d(256+N_long*N_expr, 128, kernel_size=3, stride=1, padding=1, bias=False),
                 nn.BatchNorm2d(128),
                 nn.ReLU(inplace=True)
                ]
        self.dconv3 = nn.Sequential(*dconv3) 
        
        dconv4 = [nn.Upsample(scale_factor=2),
                 nn.Conv2d(128+N_long*N_expr, 64, kernel_size=3, stride=1, padding=1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True)
                ]
        self.dconv4 = nn.Sequential(*dconv4) 
        
        dconv5 = [nn.Upsample(scale_factor=2),
                 nn.Conv2d(64+N_long*N_expr, 32, kernel_size=3, stride=1, padding=1, bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True)
                ]
        self.dconv5 = nn.Sequential(*dconv5) 
        
        dconv6 = [nn.Upsample((128, 128)),
                 nn.Conv2d(32+N_long*N_expr, 16, kernel_size=3, stride=1, padding=1),
                 nn.BatchNorm2d(16),
                 nn.ReLU(inplace=True)
                ]
        self.dconv6 = nn.Sequential(*dconv6) 
        dconv7 = [nn.Conv2d(16+N_long*N_expr, 3, kernel_size=3, stride=1, padding=1),
                 nn.Tanh()
                ]
        self.dconv7 = nn.Sequential(*dconv7)        
        
    def forward(self, input, y):
        
        x = concat_label(input, y, self.duplicate)
        x = self.fc(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.relu(x)
        x = concat_label(x, y)
        
        x = self.dconv1(x)
        x = concat_label(x, y)
        
        x = self.dconv2(x)
        x = concat_label(x, y)
        
        x = self.dconv3(x)
        x = concat_label(x, y)
        
        x = self.dconv4(x)
        x = concat_label(x, y)
        
        x = self.dconv5(x)
        x = concat_label(x, y)
        
        x = self.dconv6(x)
        x = concat_label(x, y)
        
        x_final = self.dconv7(x)
                
        return x_final
    
class Discriminator_z(nn.Module):
    def __init__(self):
        super(Discriminator_z, self).__init__()
        
        fc = [nn.Linear(50, 64),
               nn.BatchNorm1d(64),
               nn.ReLU(inplace=True),
               nn.Linear(64, 32),
               nn.BatchNorm1d(32),
               nn.ReLU(inplace=True),
               nn.Linear(32, 16),
               nn.BatchNorm1d(16),
               nn.ReLU(inplace=True),
               nn.Linear(16, 1)
                  ]
        self.fc = nn.Sequential(*fc)
        
        self.sigmoid = nn.Sigmoid()
       
    def forward(self, input):
        x = self.fc(input)
        x_final = self.sigmoid(x)
        
        return x_final, x

class Discriminator_img(nn.Module):
    def __init__(self, N_expr=6, N_long=5):
        super(Discriminator_img, self).__init__()
        
        conv1 = [nn.Conv2d(3+N_expr, 16, kernel_size=5, stride=2, padding=2),
                 nn.BatchNorm2d(16),
                 nn.ReLU(inplace=True)
                ]
        self.conv1 = nn.Sequential(*conv1)
        
        conv2 = [nn.Conv2d(16+N_expr, 32, kernel_size=5, stride=2, padding=2),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True)
                ]
        self.conv2 = nn.Sequential(*conv2)
        
        conv3 = [nn.Conv2d(32+N_expr, 64, kernel_size=5, stride=2, padding=2),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True)
                ]
        self.conv3 = nn.Sequential(*conv3)
        
        conv4 = [nn.Conv2d(64+N_expr, 128, kernel_size=5, stride=2, padding=2),
                 nn.BatchNorm2d(128),
                 nn.ReLU(inplace=True)
                ]
        self.conv4 = nn.Sequential(*conv4)
        
        fc1 = [nn.Linear((128+N_expr)*8*8, 1024),
               nn.BatchNorm1d(1024),
               nn.LeakyReLU(inplace=True)
              ]
        self.fc1 = nn.Sequential(*fc1)
        
        # -------img----------

        self.fc2 = nn.Linear(1024+N_expr, 1)
        
        self.sigmoid = nn.Sigmoid()
        
        # --------Q------------
        q_fc_shared = [nn.Linear(1024+N_expr, 128),
                       nn.BatchNorm1d(128),
                       nn.LeakyReLU(inplace=True)
                      ]
        self.q_fc_shared = nn.Sequential(*q_fc_shared)
        
        q_fc_1 = [nn.Linear(128, 64),
                  nn.BatchNorm1d(64),
                  nn.LeakyReLU(inplace=True),
                  nn.Linear(64, N_long),
                  nn.Tanh()
                 ]
        self.q_fc_1 = nn.Sequential(*q_fc_1)
        
        q_fc_2 = [nn.Linear(128, 64),
                  nn.BatchNorm1d(64),
                  nn.LeakyReLU(inplace=True),
                  nn.Linear(64, N_long),
                  nn.Tanh()
                 ]
        self.q_fc_2 = nn.Sequential(*q_fc_2)
        
        q_fc_3 = [nn.Linear(128, 64),
                  nn.BatchNorm1d(64),
                  nn.LeakyReLU(inplace=True),
                  nn.Linear(64, N_long),
                  nn.Tanh()
                 ]
        self.q_fc_3 = nn.Sequential(*q_fc_3)
        
        q_fc_4 = [nn.Linear(128, 64),
                  nn.BatchNorm1d(64),
                  nn.LeakyReLU(inplace=True),
                  nn.Linear(64, N_long),
                  nn.Tanh()
                 ]
        self.q_fc_4 = nn.Sequential(*q_fc_4)
        
        q_fc_5 = [nn.Linear(128, 64),
                  nn.BatchNorm1d(64),
                  nn.LeakyReLU(inplace=True),
                  nn.Linear(64, N_long),
                  nn.Tanh()
                 ]
        self.q_fc_5 = nn.Sequential(*q_fc_5)
        
        q_fc_6 = [nn.Linear(128, 64),
                  nn.BatchNorm1d(64),
                  nn.LeakyReLU(inplace=True),
                  nn.Linear(64, N_long),
                  nn.Tanh()
                 ]
        self.q_fc_6 = nn.Sequential(*q_fc_6)

       
    def forward(self, input, y):
        x = concat_label(input, y)
        x = self.conv1(x)
        x = concat_label(x, y)
        
        x = self.conv2(x)
        x = concat_label(x, y)
        
        x = self.conv3(x)
        x = concat_label(x, y)
        
        x = self.conv4(x)
        x = concat_label(x, y)
        x = x.view(-1, (128+6)*8*8)
        
        x = self.fc1(x)
        shared = concat_label(x, y)
        
        disc = self.fc2(shared)
        disc_sigmoid = self.sigmoid(disc)
        
        q_shared = self.q_fc_shared(shared)
        
        cat1 = self.q_fc_1(q_shared)
        cat2 = self.q_fc_2(q_shared)
        cat3 = self.q_fc_3(q_shared)
        cat4 = self.q_fc_4(q_shared)
        cat5 = self.q_fc_5(q_shared)
        cat6 = self.q_fc_6(q_shared)
                
        return disc_sigmoid, disc, q_shared, cat1, cat2, cat3, cat4, cat5, cat6