from torchvision import transforms
from PIL import Image
import os

class error(object):
    # calculate error of every epoch
    def __init__(self, dir):
        filename = 'error_record.txt'
        self.path = os.path.join(dir, filename)
        if os.path.exists(self.path):
            open(self.path, 'w').close()
        else:
            os.mknod(self.path)
    
    def initialize(self):
        self.Loss_G = 0
        self.Loss_D_z = 0
        self.Loss_D_img = 0
        self.count = 0
    
    def add(self, Loss_G, Loss_D_z, Loss_D_img):
        self.Loss_G += Loss_G
        self.Loss_D_z += Loss_D_z
        self.Loss_D_img += Loss_D_img
        self.count += 1
    
    def calculate(self):
        self.Loss_G /= self.count
        self.Loss_D_z /= self.count
        self.Loss_D_img /= self.count
    
    def print_errors(self, epoch):
        self.calculate()
        with open(self.path, 'a') as f:
            f.write('epoch{0}:\tLoss_G: {1}\tLoss_D_z: {2}\tLoss_D_img: {3}'.format(epoch, self.Loss_G, self.Loss_D_z, self.Loss_D_img))
        print('Loss_G: {0}\Loss_D_z: {1}\Loss_D_img: {2}'.format(self.Loss_G, self.Loss_D_z, self.Loss_D_img))
        return self.Loss_G, self.Loss_D_z, self.Loss_D_img