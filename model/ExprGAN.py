import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
from PIL import Image
import numpy as np
import gc
import torch.cuda
import imp

from model.Component import Encoder
from model.Component import Decoder
from model.Component import Discriminator_z
from model.Component import Discriminator_img
from model.Component import one_hot
from model.Component import one_hot_long
from model.Component import one_hot_intensity
from model.Component import weights_init_normal
from model.Component import Tensor2Image
from model.base_model import BaseModel

from model.vgg_face_dag import *

class ExprGAN(BaseModel):
    def name(self):
        return 'ExprGAN'
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.is_Train = opt.is_Train
        self.batchsize = opt.batchsize
        
        self.Encoder = Encoder()
        self.Decoder = Decoder()
        self.Discriminator_z = Discriminator_z()
        self.Discriminator_img = Discriminator_img()
        
        if self.is_Train:
            MainModel = imp.load_source('MainModel', '/ExprGAN/model/vgg_face_dag.py')
            self.face_embedding = MainModel.vgg_face_dag('ExprGAN/model/vgg_face_dag.pth')
            
            for param in self.face_embedding.parameters():
                param.requires_grad = False
            
            self.optimizer_G = optim.Adam([{'params': self.Encoder.parameters()},{'params': self.Decoder.parameters()}], 
                                          lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = optim.Adam(self.Decoder.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D_z = optim.Adam(self.Discriminator_z.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D_img = optim.Adam(self.Discriminator_img.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.CE_criterion = nn.BCEWithLogitsLoss().cuda()
            self.L1_criterion = nn.L1Loss().cuda()
            self.L2_criterion = nn.MSELoss().cuda()
        self.N_expr = opt.N_expr
            
    def init_weights(self):
        self.Encoder.apply(weights_init_normal)
        self.Decoder.apply(weights_init_normal)
        self.Discriminator_z.apply(weights_init_normal)
        self.Discriminator_img.apply(weights_init_normal)
    
    def load_input(self, input):
        self.image = []
        self.expr = []
        
        for i in input[0]:
            self.image.append(i)
        for j in input[1]:
            self.expr.append(j)
            
    def set_input(self, input):
        self.load_input(input)
        
        self.image = torch.stack(self.image, dim = 0)
        
        self.batch_expr = torch.LongTensor(self.expr)
        self.batch_expr_one_hot = one_hot(self.batch_expr, self.N_expr)
        self.batch_expr_long = one_hot_long(self.batch_expr, self.N_expr)
#         self.batch_expr = torch.LongTensor(np.random.randint(self.N_expr, size = self.batchsize))
#         self.batch_expr_one_hot = one_hot(self.batch_expr, self.N_expr)
#         self.batch_expr_long = one_hot_long(self.batch_expr, self.N_expr)
        
        self.batch_expr_long_1 = self.batch_expr_long[:, 0*5:1*5]
        self.batch_expr_long_2 = self.batch_expr_long[:, 1*5:2*5]
        self.batch_expr_long_3 = self.batch_expr_long[:, 2*5:3*5]
        self.batch_expr_long_4 = self.batch_expr_long[:, 3*5:4*5]
        self.batch_expr_long_5 = self.batch_expr_long[:, 4*5:5*5]
        self.batch_expr_long_6 = self.batch_expr_long[:, 5*5:6*5]
        
        self.z_prior = torch.Tensor(self.batchsize, 50).uniform_(-1,1)

        self.cross_ones = torch.ones(self.batchsize, 1)
        self.cross_zeros = torch.zeros(self.batchsize, 1)

        #cuda
        if self.opt.gpu_ids:
            self.image = self.image.cuda()
            
            self.batch_expr = self.batch_expr.cuda()
            self.batch_expr_one_hot = self.batch_expr_one_hot.cuda()
            self.batch_expr_long = self.batch_expr_long.cuda()
            
            self.batch_expr_long_1 = self.batch_expr_long_1.cuda()
            self.batch_expr_long_2 = self.batch_expr_long_2.cuda()
            self.batch_expr_long_3 = self.batch_expr_long_3.cuda()
            self.batch_expr_long_4 = self.batch_expr_long_4.cuda()
            self.batch_expr_long_5 = self.batch_expr_long_5.cuda()
            self.batch_expr_long_6 = self.batch_expr_long_6.cuda()
            
            self.z_prior = self.z_prior.cuda()
            
            self.cross_ones = self.cross_ones.cuda()
            self.cross_zeros = self.cross_zeros.cuda()

        self.image = Variable(self.image)
        
        self.batch_expr = Variable(self.batch_expr)
        self.batch_expr_one_hot = Variable(self.batch_expr_one_hot)# emo
        self.batch_expr_long = Variable(self.batch_expr_long)# rb
        
        self.batch_expr_long_1 = Variable(self.batch_expr_long_1)
        self.batch_expr_long_2 = Variable(self.batch_expr_long_2)
        self.batch_expr_long_3 = Variable(self.batch_expr_long_3)
        self.batch_expr_long_4 = Variable(self.batch_expr_long_4)
        self.batch_expr_long_5 = Variable(self.batch_expr_long_5)
        self.batch_expr_long_6 = Variable(self.batch_expr_long_6)
        
        self.z_prior = Variable(self.z_prior)
        
        self.cross_ones = Variable(self.cross_ones)
        self.cross_zeros = Variable(self.cross_zeros)

    def forward(self, input, stage):
        self.set_input(input)
        if stage == 1:
            self.G = self.Decoder(self.z_prior, self.batch_expr_long)
            
            self.D_G, self.D_G_logits, _, self.D_cont_G_1, self.D_cont_G_2, self.D_cont_G_3, self.D_cont_G_4, self.D_cont_G_5, self.D_cont_G_6 = self.Discriminator_img(self.G, self.batch_expr_one_hot)
            self.D_input, self.D_input_logits, _, _, _, _, _, _, _ = self.Discriminator_img(self.image, self.batch_expr_one_hot)
        elif stage == 2:
            self.z = self.Encoder(self.image)
            self.G = self.Decoder(self.z, self.batch_expr_long)
            
            self.D_G, self.D_G_logits, self.D_G_feats, self.D_cont_G_1, self.D_cont_G_2, self.D_cont_G_3, self.D_cont_G_4, self.D_cont_G_5, self.D_cont_G_6 = self.Discriminator_img(self.G, self.batch_expr_one_hot)
            self.D_input, self.D_input_logits, self.D_input_feats, self.D_cont_input_1, self.D_cont_input_2, self.D_cont_input_3, self.D_cont_input_4, self.D_cont_input_5, self.D_cont_input_6 = self.Discriminator_img(self.image, self.batch_expr_one_hot)
            
            self.real_conv1_2, self.real_conv2_2, self.real_conv3_2, self.real_conv4_2, self.real_conv5_2 = self.face_embedding(self.image)
            self.fake_conv1_2, self.fake_conv2_2, self.fake_conv3_2, self.fake_conv4_2, self.fake_conv5_2 = self.face_embedding(self.G)
        elif stage == 3:
            # 真实身份特征
            self.z = self.Encoder(self.image)
            # 生成的图片
            self.G = self.Decoder(self.z, self.batch_expr_long)

            # Dz
            self.D_z, self.D_z_logits = self.Discriminator_z(self.z)
            self.D_z_prior, self.D_z_prior_logits = self.Discriminator_z(self.z_prior)

            # 对抗
            self.D_G, self.D_G_logits, self.D_G_feats, self.D_cont_G_1, self.D_cont_G_2, self.D_cont_G_3, self.D_cont_G_4, self.D_cont_G_5, self.D_cont_G_6 = self.Discriminator_img(self.G, self.batch_expr_one_hot)
            self.D_input, self.D_input_logits, self.D_input_feats, self.D_cont_input_1, self.D_cont_input_2, self.D_cont_input_3, self.D_cont_input_4, self.D_cont_input_5, self.D_cont_input_6 = self.Discriminator_img(self.image, self.batch_expr_one_hot)
            
            self.real_conv1_2, self.real_conv2_2, self.real_conv3_2, self.real_conv4_2, self.real_conv5_2 = self.face_embedding(self.image)
            self.fake_conv1_2, self.fake_conv2_2, self.fake_conv3_2, self.fake_conv4_2, self.fake_conv5_2 = self.face_embedding(self.G)
            

           
    def backward_G(self, stage):   
        if stage == 1:
            self.G_img_loss = torch.mean(self.CE_criterion(self.D_G_logits, self.cross_ones))
            self.D_cont_loss_fake = torch.mean(torch.pow((self.D_cont_G_1 - self.batch_expr_long_1), 2)) + torch.mean(torch.pow((self.D_cont_G_2 - self.batch_expr_long_2), 2)) + torch.mean(torch.pow((self.D_cont_G_3 - self.batch_expr_long_3), 2)) + torch.mean(torch.pow((self.D_cont_G_4 - self.batch_expr_long_4), 2)) + torch.mean(torch.pow((self.D_cont_G_5 - self.batch_expr_long_5), 2)) + torch.mean(torch.pow((self.D_cont_G_6 - self.batch_expr_long_6), 2))
            
            self.loss_EG = self.G_img_loss + self.D_cont_loss_fake
        elif stage == 2:
            self.EG_loss = torch.mean(torch.abs(self.image - self.G))
            self.fm_loss = torch.mean(torch.abs(self.D_input_feats - self.D_G_feats))
            self.D_cont_loss_fake = torch.mean(torch.pow((self.D_cont_G_1 - self.batch_expr_long_1), 2)) + torch.mean(torch.pow((self.D_cont_G_2 - self.batch_expr_long_2), 2)) + torch.mean(torch.pow((self.D_cont_G_3 - self.batch_expr_long_3), 2)) + torch.mean(torch.pow((self.D_cont_G_4 - self.batch_expr_long_4), 2)) + torch.mean(torch.pow((self.D_cont_G_5 - self.batch_expr_long_5), 2)) + torch.mean(torch.pow((self.D_cont_G_6 - self.batch_expr_long_6), 2))
            
            self.conv1_2_loss = torch.mean(torch.abs(self.real_conv1_2 - self.fake_conv1_2)) / 224. / 224.
            self.conv2_2_loss = torch.mean(torch.abs(self.real_conv2_2 - self.fake_conv2_2)) / 112. / 112.
            self.conv3_2_loss = torch.mean(torch.abs(self.real_conv3_2 - self.fake_conv3_2)) / 56. / 56.
            self.conv4_2_loss = torch.mean(torch.abs(self.real_conv4_2 - self.fake_conv4_2)) / 28. / 28.
            self.conv5_2_loss = torch.mean(torch.abs(self.real_conv5_2 - self.fake_conv5_2)) / 14. / 14.
            self.vgg_loss = self.conv1_2_loss + self.conv2_2_loss + self.conv3_2_loss + self.conv4_2_loss + self.conv5_2_loss
            
            self.loss_EG = self.vgg_loss + self.EG_loss + 0 * self.fm_loss + 1 * self.D_cont_loss_fake
        elif stage == 3:
            self.EG_loss = torch.mean(torch.abs(self.image - self.G))
            self.fm_loss = torch.mean(torch.abs(self.D_input_feats - self.D_G_feats))
            self.G_img_loss = torch.mean(self.CE_criterion(self.D_G_logits, self.cross_ones))
            self.E_z_loss = torch.mean(self.CE_criterion(self.D_z_logits, self.cross_ones))
            self.tv_loss = torch.div(torch.sum(torch.pow((self.G[:,:,1:,:]-self.G[:,:,:127,:]),2)), 2.*128.*self.batchsize)+\
                      torch.div(torch.sum(torch.pow((self.G[:,:,:,1:]-self.G[:,:,:,:127]),2)), 2.*128.*self.batchsize)
            self.D_cont_loss_fake = torch.mean(torch.pow((self.D_cont_G_1 - self.batch_expr_long_1), 2)) + torch.mean(torch.pow((self.D_cont_G_2 - self.batch_expr_long_2), 2)) + torch.mean(torch.pow((self.D_cont_G_3 - self.batch_expr_long_3), 2)) + torch.mean(torch.pow((self.D_cont_G_4 - self.batch_expr_long_4), 2)) + torch.mean(torch.pow((self.D_cont_G_5 - self.batch_expr_long_5), 2)) + torch.mean(torch.pow((self.D_cont_G_6 - self.batch_expr_long_6), 2))
            
            self.conv1_2_loss = torch.mean(torch.abs(self.real_conv1_2 - self.fake_conv1_2)) / 224. / 224.
            self.conv2_2_loss = torch.mean(torch.abs(self.real_conv2_2 - self.fake_conv2_2)) / 112. / 112.
            self.conv3_2_loss = torch.mean(torch.abs(self.real_conv3_2 - self.fake_conv3_2)) / 56. / 56.
            self.conv4_2_loss = torch.mean(torch.abs(self.real_conv4_2 - self.fake_conv4_2)) / 28. / 28.
            self.conv5_2_loss = torch.mean(torch.abs(self.real_conv5_2 - self.fake_conv5_2)) / 14. / 14.
            self.vgg_loss = self.conv1_2_loss + self.conv2_2_loss + self.conv3_2_loss + self.conv4_2_loss + self.conv5_2_loss
            
            # paper weight: EG=1,vgg=1, cont=1, adv_img0.01, adv_z=0.01, tv=0.001
            self.loss_EG = self.vgg_loss + 2 * self.EG_loss + 0 * self.fm_loss + 0.01 * self.G_img_loss + 0.01 * self.E_z_loss + 0.001 * self.tv_loss + 1 * self.D_cont_loss_fake
        
        self.loss_EG.backward(retain_graph=True)
        
    def backward_D_z(self):
        self.D_z_loss_prior = torch.mean(self.CE_criterion(self.D_z_prior_logits, self.cross_ones))
        self.D_z_loss_z = torch.mean(self.CE_criterion(self.D_z_logits, self.cross_zeros))
        self.loss_Dz = self.D_z_loss_prior + self.D_z_loss_z
        self.loss_Dz.backward(retain_graph=True)
        
    def backward_D_img(self, stage):
        
        self.D_img_loss_input = torch.mean(self.CE_criterion(self.D_input_logits, self.cross_ones))
        self.D_img_loss_G = torch.mean(self.CE_criterion(self.D_G_logits, self.cross_zeros))
        self.D_cont_loss_fake = torch.mean(torch.pow((self.D_cont_G_1 - self.batch_expr_long_1), 2)) + torch.mean(torch.pow((self.D_cont_G_2 - self.batch_expr_long_2), 2)) + torch.mean(torch.pow((self.D_cont_G_3 - self.batch_expr_long_3), 2)) + torch.mean(torch.pow((self.D_cont_G_4 - self.batch_expr_long_4), 2)) + torch.mean(torch.pow((self.D_cont_G_5 - self.batch_expr_long_5), 2)) + torch.mean(torch.pow((self.D_cont_G_6 - self.batch_expr_long_6), 2))

        self.loss_Di = self.D_img_loss_input + self.D_img_loss_G + self.D_cont_loss_fake

        self.loss_Di.backward(retain_graph=True)
    
    def optimize_G_parameters(self, stage):
        if stage == 1:
            self.optimizer_D.zero_grad()
            self.backward_G(stage)
            self.optimizer_D.step()
        elif stage == 2 or stage == 3:
            self.optimizer_G.zero_grad()
            self.backward_G(stage)
            self.optimizer_G.step()
    
    def optimize_D_z_parameters(self):
        self.optimizer_D_z.zero_grad()
        self.backward_D_z()
        self.optimizer_D_z.step()
        
    def optimize_D_img_parameters(self, stage):
        self.optimizer_D_img.zero_grad()
        self.backward_D_img(stage)
        self.optimizer_D_img.step()

    def print_current_error(self):
        print('loss G: {0} \t loss D_z: {1} \t loss D_img: {2}'.format(self.loss_EG.data[0], self.loss_Dz.data[0], self.loss_Di.data[0]))
    
    def save(self, epoch):
        self.save_network(self.Encoder, 'Encoder', epoch, self.gpu_ids)
        self.save_network(self.Decoder, 'Decoder', epoch, self.gpu_ids)
        self.save_network(self.Discriminator_z, 'D_z', epoch, self.gpu_ids)
        self.save_network(self.Discriminator_img, 'D_img', epoch, self.gpu_ids)
    
    def save_result(self, epoch=None):
        for i, syn_img in enumerate(self.G.data):
            img = self.image.data[i]
            expr = self.batch_expr.data[i].item()
            filename = 'image_' + str(i) + '_' + str(expr) + '.png'

            if epoch:
                filename = 'epoch{0}_{1}'.format(epoch, filename)

            path = os.path.join('/ExprGAN/image', filename)
                
            img = Tensor2Image(img)
            syn_img = Tensor2Image(syn_img)

            width, height = img.size
            result_img = Image.new(img.mode, (width*2, height))
            result_img.paste(img, (0, 0, width, height))
            result_img.paste(syn_img, box=(width, 0))
            result_img.save(path, quality=95)