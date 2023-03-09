import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import lpips
from prep import printProgressBar
from model.generator import Generator
from model.discriminator import Discriminator
from model.charbonnier_loss import CharbonnierLoss
from measure import compute_measure
from loader import *
from torch.optim import lr_scheduler


class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max
        self.validate_path = args.validate_path
        self.save_path = args.save_path
        self.validate_patient = args.validate_patient
        self.test_patient = args.test_patient
        self.multi_gpu = args.multi_gpu
        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.n_d_train = args.n_d_train

        self.patch_n = args.patch_n
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size

        self.lr = args.lr

        self.netG = Generator()
        self.netD = Discriminator()

        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.netG = nn.DataParallel(self.netG)
            self.netD = nn.DataParallel(self.netD)
        # torch.backends.cudnn.enabled = False
        self.netG.to(self.device)
        self.netD.to(self.device)

        self.lpips_loss = lpips.LPIPS(net='vgg', verbose=False).to(self.device)
        self.cbLoss = CharbonnierLoss().to(self.device)
        self.optimizer_G = optim.AdamW(self.netG.parameters(), self.lr, betas=(0.9, 0.999))
        self.optimizer_D = optim.AdamW(self.netD.parameters(), self.lr, betas=(0.9, 0.999))
        self.scheduler_G = lr_scheduler.StepLR(self.optimizer_G, step_size=self.decay_iters, gamma=0.5)
        self.scheduler_D = lr_scheduler.StepLR(self.optimizer_D, step_size=self.decay_iters, gamma=0.5)

    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'Generator_{}iter.ckpt'.format(iter_))
        torch.save(self.netG.state_dict(), f)

    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'Generator_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.netG.load_state_dict(state_d)
        else:
            self.netG.load_state_dict(torch.load(f))

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self, z, x, y, pred, fig_name, original_result, pred_result):
        z, x, y, pred = z.numpy(), x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 4, figsize=(30, 10))
        ax[0].imshow(z, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Content', fontsize=30)

        ax[1].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('HQ', fontsize=30)

        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('LQ', fontsize=30)
        ax[2].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[3].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[3].set_title('fake_LQ', fontsize=30)
        ax[3].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def train(self):
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0.0, 0.0, 0.0
        netG_param = self.get_parameter_number(self.netG)
        netD_param = self.get_parameter_number(self.netD)
        print('netG_param=', netG_param)
        print('netD_param=', netD_param)
        # Training
        print('START TRAINING...')
        total_iters = 0
        start_time = time.time()
        for epoch in range(1, self.num_epochs):
            for iter_, (HQ, LQ, SHQ) in enumerate(self.data_loader):
                total_iters += 1
                # add 1 channel
                HQ = HQ.unsqueeze(0).float().to(self.device)
                LQ = LQ.unsqueeze(0).float().to(self.device)
                SHQ = SHQ.unsqueeze(0).float().to(self.device)
                # patch training
                if self.patch_size:
                    HQ = HQ.view(-1, 1, self.patch_size, self.patch_size)
                    LQ = LQ.view(-1, 1, self.patch_size, self.patch_size)
                    SHQ = SHQ.view(-1, 1, self.patch_size, self.patch_size)

                # Discriminator
                self.optimizer_D.zero_grad()
                self.netD.zero_grad()
                pred = self.netG(SHQ, HQ)
                real_scores = self.netD(LQ)
                fake_scores = self.netD(pred.detach())
                loss_D = -torch.mean(real_scores) + torch.mean(fake_scores)
                loss_D.backward()
                self.optimizer_D.step()

                # generator, perceptual loss
                self.optimizer_G.zero_grad()
                self.netG.zero_grad()
                pred = self.netG(SHQ, HQ)
                perceptual_loss = self.lpips_loss(LQ, pred)
                self.fake_scores = self.netD(pred)
                loss_G = -torch.mean(self.fake_scores) + self.cbLoss(LQ, pred) + 0.5 * perceptual_loss
                loss_G.backward()
                self.optimizer_G.step()

                # print
                if total_iters % self.print_iters == 0:
                    print(
                        "STEP [{}], EPOCH [{}/{}], ITER [{}/{}], TIME [{:.1f}s]\nG_LOSS: {:.8f}, D_LOSS: {:.8f}".format(
                            total_iters, epoch, self.num_epochs, iter_ + 1, len(self.data_loader),
                            time.time() - start_time, loss_G.item(), loss_D.item()))
                # learning rate decay
                self.scheduler_D.step()
                self.scheduler_G.step()
                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)

            # Validation
            if total_iters % self.save_iters == 0 and total_iters != 0:
                validate_data_loader = get_loader(mode='test',
                             load_mode=0,
                             saved_path=self.validate_path,
                             test_patient=self.validate_patient,
                             patch_n=(None),
                             patch_size=(None),
                             transform=False,
                             batch_size=1,
                             num_workers=8)
                self.netG.eval()
                with torch.no_grad():
                    for i, (HQ, LQ, SHQ) in enumerate(validate_data_loader):
                        HQ = HQ.unsqueeze(0).float().to(self.device)
                        LQ = LQ.unsqueeze(0).float().to(self.device)
                        SHQ = SHQ.unsqueeze(0).float().to(self.device)
                        pred = self.netG(SHQ, HQ)
                        HQ1 = self.trunc(self.denormalize_(HQ))
                        LQ1 = self.trunc(self.denormalize_(LQ))
                        pred1 = self.trunc(self.denormalize_(pred))
                        data_range = self.trunc_max - self.trunc_min
                        _, pred_result = compute_measure(LQ1, HQ1, pred1, data_range)

                        pred_psnr_avg += pred_result[0]
                        pred_ssim_avg += pred_result[1]
                        pred_rmse_avg += pred_result[2]

                #########################################################
                # 日志文件
                with open('pred_psnr_avg.txt', 'a') as f:
                    f.write('EPOCH:%d pred_psnr:%.20f' % (epoch, pred_psnr_avg / len(validate_data_loader)) + '\n')
                    f.close()

                with open('pred_ssim_avg.txt', 'a') as f:
                    f.write('EPOCH:%d pred_ssim:%.20f' % (epoch, pred_ssim_avg / len(validate_data_loader)) + '\n')
                    f.close()

                with open('pred_rmse_avg.txt', 'a') as f:
                    f.write('EPOCH:%d pred_rmse:%.20f' % (epoch, pred_rmse_avg / len(validate_data_loader)) + '\n')
                    f.close()

                pred_psnr_avg = 0
                pred_ssim_avg = 0
                pred_rmse_avg = 0
            #########################################################

            else:
                continue
        print('TUNNING DONE')

    def test(self):
        # load model
        self.netG = Generator().to(self.device)
        self.load_model(self.test_iters)
        self.netG.eval()
        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        ori_psnr_avg1, ori_ssim_avg1, ori_rmse_avg1 = [], [], []
        pred_psnr_avg1, pred_ssim_avg1, pred_rmse_avg1 = [], [], []

        with torch.no_grad():

            for i, (HQ, LQ, SHQ) in enumerate(self.data_loader):
                shape_ = LQ.shape[-1]
                HQ = HQ.unsqueeze(0).float().to(self.device)
                LQ = LQ.unsqueeze(0).float().to(self.device)
                SHQ = SHQ.unsqueeze(0).float().to(self.device)
                pred = self.netG(SHQ, HQ)
                # denormalize, truncate
                HQ1 = self.trunc(self.denormalize_(HQ.view(shape_, shape_).cpu().detach())).double()
                np.save(os.path.join(self.save_path, 'HQ', '{}_result'.format(i)), HQ1)
                SHQ1 = self.trunc(self.denormalize_(SHQ.view(shape_, shape_).cpu().detach())).double()
                np.save(os.path.join(self.save_path, 'SHQ', '{}_result'.format(i)), SHQ1)
                LQ1 = self.trunc(self.denormalize_(LQ.view(shape_, shape_).cpu().detach())).double()
                np.save(os.path.join(self.save_path, 'LQ', '{}_result'.format(i)), LQ1)
                pred1 = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach())).double()
                np.save(os.path.join(self.save_path, 'pred', '{}_result'.format(i)), pred1)

                data_range = self.trunc_max - self.trunc_min

                original_result1, pred_result1 = compute_measure(LQ, HQ, pred, data_range)
                ori_psnr_avg += original_result1[0]
                ori_psnr_avg1.append(ori_psnr_avg / len(self.data_loader))
                ori_ssim_avg += original_result1[1]
                ori_ssim_avg1.append(ori_ssim_avg / len(self.data_loader))
                ori_rmse_avg += original_result1[2]
                ori_rmse_avg1.append(ori_rmse_avg / len(self.data_loader))
                pred_psnr_avg += pred_result1[0]
                pred_psnr_avg1.append(pred_psnr_avg / len(self.data_loader))
                pred_ssim_avg += pred_result1[1]
                pred_ssim_avg1.append(pred_ssim_avg / len(self.data_loader))
                pred_rmse_avg += pred_result1[2]
                pred_rmse_avg1.append(pred_rmse_avg / len(self.data_loader))

                # save result figure
                if self.result_fig:
                    self.save_fig(SHQ, HQ, LQ, pred, i, original_result1, pred_result1)

            print('\n')
            print('Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                ori_psnr_avg / len(self.data_loader), ori_ssim_avg / len(self.data_loader),
                ori_rmse_avg / len(self.data_loader)))
            print('After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                pred_psnr_avg / len(self.data_loader), pred_ssim_avg / len(self.data_loader),
                pred_rmse_avg / len(self.data_loader)))
