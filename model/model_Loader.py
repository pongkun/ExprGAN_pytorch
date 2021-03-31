import torch
from model.ExprGAN import ExprGAN

def CreatModel(opt):
    model = ExprGAN()
    
    model.initialize(opt)
    model.init_weights()
    
    if opt.is_Train and opt.load_epoch:
        model.reload(opt.load_epoch)
        print('Pretrain models were loaded, epoch is {}'.format(opt.load_epoch))

    if len(opt.gpu_ids) and torch.cuda.is_available():
        model.face_embedding.cuda()
        model.Encoder.cuda()
        model.Decoder.cuda()
        model.Discriminator_z.cuda()
        model.Discriminator_img.cuda()

        model.L1_criterion.cuda()
        model.L2_criterion.cuda()
        model.CE_criterion.cuda()

    print('model {} was created'.format(model.name()))
    return model

def CreatModel_test(opt, load_epoch):
    model = ExprGAN()
    
    model.initialize(opt)
    model.init_weights()
    
    model.reload(load_epoch)
    print('Pretrain models were loaded, epoch is {}'.format(load_epoch))

    if len(opt.gpu_ids) and torch.cuda.is_available():
        model.Encoder.cuda()
        model.Decoder.cuda()

    print('model {} was created'.format(model.name()))
    return model