import os
import torch

class BaseModel(object):
    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_Train = opt.is_Train
        self.save_dir = opt.checkpoints_dir
        self.load_dir = opt.test_checkpoints_dir
        self.result_dir = os.path.join(opt.test_dir, opt.pretrained_G) if opt.pretrained_G else opt.test_dir
    
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '{0}_net_{1}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    def load(self, network, filename):
        save_path = os.path.join(self.load_dir, filename)
        network.load_state_dict(torch.load(save_path))

    def reload(self, count_epoch):
        Encoder_filename = '{}_net_Encoder.pth'.format(count_epoch)
        Decoder_filename = '{}_net_Decoder.pth'.format(count_epoch)
        D_z_filename = '{}_net_D_z.pth'.format(count_epoch)
        D_img_filename = '{}_net_D_img.pth'.format(count_epoch)
        self.load(self.Encoder, Encoder_filename)
        self.load(self.Decoder, Decoder_filename)
        self.load(self.Discriminator_z, D_z_filename)
        self.load(self.Discriminator_img, D_img_filename)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f'%lr)

