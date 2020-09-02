from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as func
import math
import util
import sys
import classifier
import classifier2
import model
from sklearn import preprocessing
import scipy.io as sio
import visdom

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='dataset')
parser.add_argument('--dataroot', default='./data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=300, help='number features to generate per class')
parser.add_argument('--gzsl', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', default=False)
parser.add_argument('--validation', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=312, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.01, help='weight of the classification loss')
parser.add_argument('--gamma', type=float, default=0.1, help='weight of the regression loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='MLP_G')
parser.add_argument('--netD_name', default='MLP_CRITIC')
parser.add_argument('--save_path', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--save_name', default='CUB', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=50)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=3483, help='manual seed') 
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
#
#parser.add_argument('--fc1_size', type=int, default=4096, help='the first FC layer')
#parser.add_argument('--fc2_size', type=int, default=512, help='the second FC layer')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.save_path)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# cudnn.benchmark may help increase the running time.
cudnn.benchmark = True

ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data, this is a class instance
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# image encoder
netG_Img = model.MLP_G_Img_Adaptive(opt)
netG_Img.apply(model.weights_init)
print(netG_Img)

# generator
netG_Att = model.MLP_G_Att_Adaptive(opt)
netG_Att.apply(model.weights_init)
print(netG_Att)

# regressor (decoder)
netE = model.MLP_E_Adaptive(opt)
netE.apply(model.weights_init)
print(netE)

# discriminator
netD = model.MLP_D(opt)
netD.apply(model.weights_init)
print(netD)

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()
mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()
sim_criterion = nn.CosineEmbeddingLoss()
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)
noise_cls = torch.FloatTensor(opt.nclass_all, opt.nz)
noise_unseen = torch.FloatTensor(data.ntest_class, opt.nz)
unseen_att = torch.FloatTensor(data.ntest_class, opt.attSize)
unseen_label = torch.LongTensor(data.ntest_class)
one_batch = torch.ones(opt.batch_size).float()
one_batch_unseen = torch.ones(data.ntest_class).long()
zero_batch = torch.zeros(opt.batch_size).long()
fixed_noise = torch.FloatTensor(opt.syn_num, opt.nz).normal_(0, 1)

if opt.cuda:
    netD.cuda()
    netG_Img.cuda()
    netG_Att.cuda()
    netE.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    mse_criterion.cuda()
    sim_criterion.cuda()
    l1_criterion.cuda()
    input_label = input_label.cuda()
    noise_cls = noise_cls.cuda()
    noise_unseen = noise_unseen.cuda()
    unseen_att = unseen_att.cuda()
    unseen_label = unseen_label.cuda()
    one_batch = one_batch.cuda()
    zero_batch = zero_batch.cuda()
    one_batch_unseen = one_batch_unseen.cuda()
    fixed_noise = fixed_noise.cuda()

if opt.standardization:
    print('standardization...')
    scaler = preprocessing.StandardScaler()
else:
    scaler = preprocessing.MinMaxScaler()

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    #syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        #syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        #syn_noise.normal_(0, 1)

        with torch.no_grad():
            output = netG(fixed_noise, syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label

def compute_sim_loss(img_embed, text_embed):
    sim_loss = sim_criterion(img_embed, text_embed, one_batch)

    return sim_loss

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

print("Starting to train classifier...")
# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(data, data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                     data.seenclasses.size(0), opt.resSize, opt.cuda, opt.classifier_lr, 0.5, 100, 100,
                                     opt.pretrain_classifier)
# for p in pretrain_cls.model.parameters():  # set requires_grad to False
#     p.requires_grad = False

print("Starting to train loop...")
# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(list(netG_Img.parameters())+list(netG_Att.parameters())+
                        list(pretrain_cls.model.parameters())+list(netE.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizerU = optim.Adam(list(netG_Att.parameters())+list(netC.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
# visdom setup
#vis = util.Visualizer(env='LatGAN')
###
best_acc_zsl = 0.0
best_acc_unseen_gzsl = 0.0
best_acc_seen_gzsl = 0.0
best_H_gzsl = 0.0
opt.n_batch = data.ntrain // opt.batch_size
# freeze the classifier during the optimization

for epoch in range(opt.nepoch):
    ## random shuffle
    idx = torch.randperm(data.ntrain)
    data.train_feature = data.train_feature[idx]
    data.train_label = data.train_label[idx]
    ##
    for i in range(opt.n_batch):
        start_i = i * opt.batch_size
        end_i = start_i + opt.batch_size
        batch_feature = data.train_feature[start_i:end_i]
        batch_label = data.train_label[start_i:end_i]
        batch_att = data.attribute[batch_label]
        # copy data to GPU
        input_res.copy_(batch_feature)
        input_att.copy_(batch_att)
        input_label.copy_(util.map_label(batch_label, data.seenclasses))
        ############################
        # (1) Update D network
        ###########################
        # for p in netD.parameters():  # reset requires_grad
        #     p.requires_grad = True  # they are set to False below in netG update
        for iter_d in range(opt.critic_iter):
            ## call a function!!!
            # sample()
            netD.zero_grad()
            # train with realG
            real_res = netG_Img(input_res)
            criticD_real = netD(real_res, input_att).view(-1)
            criticD_real = criticD_real.mean()
            # criticD_real.backward(mone)
            # train with fakeG
            noise.normal_(0, 1)
            fake_res = netG_Att(noise, input_att)
            criticD_fake = netD(fake_res, input_att)  ## detach???
            criticD_fake = criticD_fake.mean()
            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_res.data, fake_res.data, input_att)
            ##
            D_gan_loss = criticD_fake - criticD_real
            D_loss = D_gan_loss + gradient_penalty
            D_loss.backward()
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        netG_Img.zero_grad()
        netG_Att.zero_grad()
        netE.zero_grad()
        pretrain_cls.model.zero_grad()
        ###
        real_res = netG_Img(input_res)
        noise.normal_(0, 1)
        fake_res = netG_Att(noise, input_att)
        criticG_fake = netD(fake_res, input_att)
        criticG_fake = criticG_fake.mean()
        G_gan_loss = -criticG_fake ### ??? remove real?
        # classification loss
        G_cls_loss_att = cls_criterion(pretrain_cls.model(fake_res), input_label)
        G_cls_loss_img = cls_criterion(pretrain_cls.model(real_res), input_label)
        G_cls_loss = G_cls_loss_att + G_cls_loss_img
        # reconst loss
        #real_res = netG_Img(input_res)
        real_rect = netE(real_res)
        G_rect_loss_img = l1_criterion(real_rect, input_att)
        fake_rect = netE(fake_res)
        G_rect_loss_att = l1_criterion(fake_rect, input_att)

        # total loss
        G_loss = G_gan_loss + opt.cls_weight * G_cls_loss + opt.gamma * G_rect_loss_att + G_rect_loss_img
        G_loss.backward()
        optimizerG.step()

    # vis.plot_loss({'D_gan_loss': D_gan_loss.item(),
    #                'G_gan_loss': G_gan_loss.item(), 'G_cls_loss_att': G_cls_loss_att.item(), 'G_cls_loss_img': G_cls_loss_img.item(),
    #                'G_rect_loss_att': G_rect_loss_att.item()},
    #                 title='LatGAN', xlabel='Epochs', ylabel='Training loss')

    #evaluate the model, set G to evaluation mode
    netG_Img.eval()
    netG_Att.eval()
    #Generalized zero-shot learning
    if opt.gzsl:
        ## test unseen img embed
        ntest = data.ntest_unseen
        nclass = data.ntest_class
        test_unseen_img_embed = torch.FloatTensor(ntest, opt.resSize)
        # test_text_embed = torch.FloatTensor(nclass, opt.embedSize)
        start = 0
        for ii in range(0, ntest, opt.batch_size):
            end = min(ntest, start + opt.batch_size)
            test_feature = data.test_unseen_feature[start:end]
            if opt.cuda:
                test_feature = test_feature.cuda()
            img_embed = netG_Img(test_feature)
            test_unseen_img_embed[start:end, :] = img_embed.data.cpu()
            start = end

        ## synthesize unseen image embed
        syn_feature, syn_label = generate_syn_feature(netG_Att, data.unseenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)

        train_X = torch.cat((train_X, syn_feature), 0)
        train_Y = torch.cat((train_Y, syn_label), 0)
        nclass = opt.nclass_all
        cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 50, opt.syn_num,
                                     True, test_unseen_feature=test_unseen_img_embed, test_seen_feature=data.test_seen_feature)
        print('%.4f %.4f %.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
        if cls.H > best_H_gzsl:
            best_H_gzsl = cls.H
            best_acc_unseen_gzsl = cls.acc_unseen
            best_acc_seen_gzsl = cls.acc_seen

    # Zero-shot learning
    else:
        syn_feature, syn_label = generate_syn_feature(netG_Att, data.unseenclasses, data.attribute, opt.syn_num)
        ## extract visual latent features
        # extract image embed for unseen classes
        ntest = data.ntest_unseen
        nclass = data.ntest_class
        test_img_embed = torch.FloatTensor(ntest, opt.resSize)
        # test_text_embed = torch.FloatTensor(nclass, opt.embedSize)
        start = 0
        for ii in range(0, ntest, opt.batch_size):
            end = min(ntest, start + opt.batch_size)
            test_feature = data.test_unseen_feature[start:end]
            if opt.cuda:
                test_feature = test_feature.cuda()
            img_embed = netG_Img(test_feature)
            test_img_embed[start:end, :] = img_embed.data.cpu()
            start = end

        cls = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data,
                                     data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 50, opt.syn_num,
                                     False, test_unseen_feature=test_img_embed, test_seen_feature=None) # opt.syn_num
        acc = cls.acc
        #print('unseen class accuracy= ', acc)
        print('[%d/%d] unseen class accuracy = %.4f' % (epoch, opt.nepoch, acc))
        #vis.plot_acc({'test unseen': acc}, title='debug', xlabel='Epochs', ylabel='Top-1 accuracy')
        if acc > best_acc_zsl:
            best_acc_zsl = acc
            print('Save model!')
            torch.save(netG_Img.state_dict(),
                        os.path.join(opt.save_path, opt.save_name + '_AVSE:netG_Img_epoch_%d.ckpt'%(epoch)))
            torch.save(netG_Att.state_dict(),
                        os.path.join(opt.save_path, opt.save_name + '_AVSE:netG_Att_epoch_%d.ckpt'%(epoch)))
            torch.save(netD.state_dict(),
                        os.path.join(opt.save_path, opt.save_name + '_AVSE:netD_epoch_%d.ckpt'%(epoch)))
            torch.save(netE.state_dict(),
                        os.path.join(opt.save_path, opt.save_name + '_AVSE:netE_epoch_%d.ckpt'%(epoch)))

    # reset G to training mode
    netG_Img.train()
    netG_Att.train()

if opt.gzsl:
    print('GZSL best unseen=%.4f, seen=%.4f, h=%.4f' % (best_acc_unseen_gzsl, best_acc_seen_gzsl, best_H_gzsl))
else:
    print('ZSL best unseen=%.4f' % (best_acc_zsl))


