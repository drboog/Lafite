# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
import torch.nn.functional as F
import torchvision.transforms as T
import clip
import dnnlib
import random
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, real_features): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, G_mani, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.G_mani = G_mani
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        clip_model, _ = clip.load("ViT-B/32", device=device)  # Load CLIP model here
        self.clip_model = clip_model.eval()
        # use a pre-trained VGG net for image-image contrastive loss ?
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            self.vgg16 = torch.jit.load(f).eval().to(device)
        
        
    def run_G(self, z, c, sync, txt_fts=None, ):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            
#             if self.style_mixing_prob > 0:
#                 new_ws = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)
#                 with torch.autograd.profiler.record_function('style_mixing'):
#                     cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
#                     cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
#                     ws[:, cutoff:] = new_ws[:, cutoff:]
            
#             if self.G_mani is not None:
#                 if txt_fts is None:
#                     txt_fts = torch.randn(z.size()[0], self.G_mani.f_dim).to(ws.device)
#                     txt_fts = txt_fts/txt_fts.norm(dim=-1, keepdim=True)
#                 ws = self.G_mani(z=txt_fts, c=c, w=ws)
                
            if self.style_mixing_prob > 0:
                new_ws = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)
#                 if self.G_mani is not None:
#                     if txt_fts is None:
#                         txt_fts = torch.randn(z.size()[0], self.G_mani.f_dim).to(ws.device)
#                         txt_fts = txt_fts/txt_fts.norm(dim=-1, keepdim=True)
#                     new_ws = self.G_mani(z=txt_fts, c=c, w=new_ws, skip_w_avg_update=True)
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = new_ws[:, cutoff:]
                
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws, fts=txt_fts)
        return img, ws

    def run_D(self, img, c, sync, fts=None):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits, d_fts = self.D(img, c, fts=fts)
        return logits, d_fts
    
    def normalize(self):
        return T.Compose([
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])    
    
    def full_preprocess(self, img, mode='bicubic'):
        full_size = img.shape[-2]

        if full_size < 224:
#             cut_size = torch.randint(9*full_size//10, full_size, ())
#             left = torch.randint(0, full_size-cut_size, ())
#             top = torch.randint(0, full_size-cut_size, ())
#             cropped_img = img[:, :, top:top+cut_size, left:left+cut_size]
#             reshaped_img = F.interpolate(cropped_img, (224, 224), mode=mode, align_corners=False)
            
            pad_1 = torch.randint(0, 224-full_size, ())
            pad_2 = torch.randint(0, 224-full_size, ())
            m = torch.nn.ConstantPad2d((pad_1, 224-full_size-pad_1, pad_2, 224-full_size-pad_2), 1.)
            reshaped_img = m(img)
        else:
            cut_size = torch.randint(full_size//2, full_size, ())
            left = torch.randint(0, full_size-cut_size, ())
            top = torch.randint(0, full_size-cut_size, ())
            cropped_img = img[:, :, top:top+cut_size, left:left+cut_size]
            reshaped_img = F.interpolate(cropped_img, (224, 224), mode=mode, align_corners=False)
        reshaped_img = (reshaped_img + 1.)*0.5 # range in [0., 1.] now
        reshaped_img = self.normalize()(reshaped_img)
        return  reshaped_img

    def custom_preprocess(self, img, ind, cut_num, mode='bicubic'):   # more to be implemented here
        full_size = img.shape[-2]
        
        grid = np.sqrt(cut_num)
        most_right = min(int((ind%grid + 1)*full_size/grid), full_size)
        most_bottom = min(int((ind//grid + 1)*full_size/grid), full_size)
        
        cut_size = torch.randint(int(full_size//(grid+1)), int(min(min(full_size//2, most_right), most_bottom)), ()) # TODO: tune this later
        left = torch.randint(0, most_right-cut_size, ())
        top = torch.randint(0, most_bottom-cut_size, ())
        cropped_img = img[:, :, top:top+cut_size, left:left+cut_size]
        reshaped_img = F.interpolate(cropped_img, (224, 224), mode=mode, align_corners=False)


        reshaped_img = (reshaped_img + 1.)*0.5 # range in [0., 1.] now
        
        reshaped_img = self.normalize()(reshaped_img)

        return  reshaped_img

    def contra_loss(self, temp, mat1, mat2, lam):
        sim = torch.cosine_similarity(mat1.unsqueeze(1), mat2.unsqueeze(0), dim=-1)
        if temp > 0.:
            sim = torch.exp(sim/temp)
            sim1 = torch.diagonal(F.softmax(sim, dim=1))*temp
            sim2 = torch.diagonal(F.softmax(sim, dim=0))*temp
            if 0.<lam < 1.:
                return lam*torch.log(sim1) + (1.-lam)*torch.log(sim2)
            elif lam == 0:
                return torch.log(sim2)
            else:
                return torch.log(sim1)
        else:
            return torch.diagonal(sim)

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, img_fts, txt_fts, lam, temp, gather, d_use_fts, itd, itc, iid, iic, mixing_prob=0., finetune=False):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        
        # augmentation
        aug_level_1 = 0.1
        aug_level_2 = 0.5
#         print(torch.cosine_similarity(img_fts, txt_fts, dim=-1))
        
        # the semantic  similarity of perturbed feature with real feature would be:
        # sim >= (sqrt(1 - aug_level^2)-aug_level)/(sqrt(1 + 2*aug_level*sqrt(1 - aug_level^2)))
        mixing_prob = mixing_prob # probability to use img_fts instead of txt_fts
        random_noise = torch.randn(img_fts.shape).to(img_fts.device)
        random_noise = random_noise/random_noise.norm(dim=-1, keepdim=True)
        txt_fts_ = txt_fts*(1-aug_level_1) + random_noise*aug_level_1
        txt_fts_ = txt_fts_/txt_fts_.norm(dim=-1, keepdim=True)
        img_fts_ = img_fts*(1-aug_level_2) + random_noise*aug_level_2
        img_fts_ = img_fts_/img_fts_.norm(dim=-1, keepdim=True)
        if mixing_prob > 0.99:
            txt_fts_ = img_fts_
        elif mixing_prob < 0.01:
            txt_fts_ = txt_fts_
        else:
            txt_fts_ = torch.where(torch.rand([txt_fts_.shape[0], 1], device=txt_fts_.device) < mixing_prob, img_fts_, txt_fts_)

            
        img_img_d = iid # discriminator
        img_img_c = iic  # clip
        img_txt_d = itd # discriminator
        img_txt_c = itc # clip
        img_img_region = 0.
        temp = temp
        lam = lam
        
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        
        def gather_tensor(input_tensor, gather_or_not):
            if gather_or_not:
                output_tensor = [torch.zeros_like(input_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(output_tensor, input_tensor)
    #           # print(torch.cat(output_tensor).size())
                return torch.cat(output_tensor)
            else:
                return input_tensor
        
        txt_fts_all = gather_tensor(txt_fts_, gather)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, txt_fts=txt_fts_, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits, gen_d_fts = self.run_D(gen_img, gen_c, sync=False, fts=txt_fts_)
                
                gen_d_fts_all = gather_tensor(gen_d_fts, gather)
                
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                if finetune:
                    loss_Gmain = 0.01*torch.square(self.vgg16((gen_img+1.)*127.5, return_lpips=True, resize_images=False) - self.vgg16((real_img+1.)*127.5, return_lpips=True, resize_images=False)).sum(-1).mean()
                else:
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))

                normed_gen_full_img = self.full_preprocess(gen_img)
                img_fts_gen_full = self.clip_model.encode_image(normed_gen_full_img)
                img_fts_gen_full = img_fts_gen_full/img_fts_gen_full.norm(dim=-1, keepdim=True)
              
                    
                img_fts_gen_full_all = gather_tensor(img_fts_gen_full, gather)
                img_fts_all = gather_tensor(img_fts, gather)
                if img_txt_c > 0.:
                    clip_loss_img_txt = self.contra_loss(temp, img_fts_gen_full_all, txt_fts_all, lam)
                    loss_Gmain = loss_Gmain - img_txt_c*clip_loss_img_txt.mean()
                    
                if img_img_c > 0.:
                    clip_loss_img_img = self.contra_loss(temp, img_fts_gen_full_all, img_fts_all, lam)
                    loss_Gmain = loss_Gmain - img_img_c*clip_loss_img_img.mean()
                    
                if img_img_region > 0.:
                    cut_num = 9
                    gen_img_fts_region = []
                    real_img_fts_region = []
                    with torch.no_grad():
                        for ind in range(cut_num):
                            real_normed_img = self.custom_preprocess(real_img, ind, cut_num)
                            new_fts_region = self.clip_model.encode_image(real_normed_img)
                            new_fts_region = new_fts_region/new_fts_region.norm(dim=-1, keepdim=True)
                            real_img_fts_region.append(new_fts_region.unsqueeze(1))
                        real_img_fts_region = torch.cat(real_img_fts_region, dim=1)
                        assert real_img_fts_region.shape[1] == cut_num
                        
                    for ind in range(cut_num):
                        gen_normed_img = self.custom_preprocess(gen_img, ind, cut_num)
                        new_fts_region = self.clip_model.encode_image(gen_normed_img)
                        new_fts_region = new_fts_region/new_fts_region.norm(dim=-1, keepdim=True)
                        gen_img_fts_region.append(new_fts_region.unsqueeze(1))
                    gen_img_fts_region = torch.cat(gen_img_fts_region, dim=1)
                    assert gen_img_fts_region.shape[1] == cut_num # n,d,512
                    
                    
                    gen_img_fts_region_all = gather_tensor(gen_img_fts_region, gather)
                    real_img_fts_region_all = gather_tensor(real_img_fts_region, gather)
                    
                    rho_1 = 5.
                    rho_2 = 5.
                    rho_3 = 50.
                    attention = torch.cosine_similarity(gen_img_fts_region_all.unsqueeze(1).unsqueeze(2), real_img_fts_region_all.unsqueeze(0).unsqueeze(3), dim=-1) # n,n,d,d
                    attention = F.softmax(attention*rho_1, dim=-1).unsqueeze(-1) #n,n,d,d,1
                    
                    reweighted_gen_img_fts_region = (attention * gen_img_fts_region_all.unsqueeze(0).unsqueeze(2)).mean(-2) #n,n,d,512
                    clip_loss_region = torch.cosine_similarity(reweighted_gen_img_fts_region, real_img_fts_region_all.unsqueeze(1), dim=-1) # n,n,d
                    clip_loss_region = torch.log(torch.sum(torch.exp(clip_loss_region*rho_2), dim=-1))/rho_2 # (n, n)
                    clip_loss_region_ = torch.diagonal(torch.log(F.softmax(clip_loss_region*rho_3, dim=1)+1e-7))/rho_3
                    loss_Gmain = loss_Gmain - img_img_region*clip_loss_region_.mean()
                    
#                     reweighted_gen_img_fts_region = (attention * gen_img_fts_region_all.unsqueeze(1).unsqueeze(2)).mean(-2) #n,n,d,512
#                     clip_loss_region = torch.cosine_similarity(reweighted_gen_img_fts_region, real_img_fts_region_all.unsqueeze(0), dim=-1) # n,n,d
#                     clip_loss_region = torch.log(torch.sum(torch.exp(clip_loss_region*rho_2), dim=-1))/rho_2 # (n, n)
                    clip_loss_region_ = torch.diagonal(torch.log(F.softmax(clip_loss_region*rho_3, dim=0)+1e-7))/rho_3
                    loss_Gmain = loss_Gmain - img_img_region*clip_loss_region_.mean()
                    
                if img_txt_d > 0.:
                    loss_Gmain = loss_Gmain - img_txt_d*self.contra_loss(temp, gen_d_fts_all, txt_fts_all, lam).mean()
                if img_img_d > 0.:
                    with torch.no_grad():
                        _, g_real_d_fts = self.run_D(real_img.detach(), real_c, sync=False, fts=txt_fts_)
                    g_real_d_fts_all = gather_tensor(g_real_d_fts, gather)
                    loss_Gmain = loss_Gmain - img_img_d*self.contra_loss(temp, g_real_d_fts_all, gen_d_fts_all, lam).mean()

                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                txt_fts_0 = txt_fts_[:batch_size]
                txt_fts_0.requires_grad_()
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], txt_fts=txt_fts_0, sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    if d_use_fts:
                        pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws, txt_fts_0], create_graph=True, only_inputs=True)[0]
                    else:
                         pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, txt_fts=txt_fts_, sync=False)
                gen_logits, gen_d_fts = self.run_D(gen_img, gen_c, sync=False, fts=txt_fts_) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits, real_d_fts = self.run_D(real_img_tmp, real_c, sync=sync, fts=txt_fts_)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    if img_txt_d > 0.:
                        real_d_fts_all = gather_tensor(real_d_fts, gather)
                        loss_Dreal = loss_Dreal - img_txt_d*self.contra_loss(temp, real_d_fts_all, txt_fts_all, lam).mean()
                    
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

# ----------------------------------------------------------------------------


class FinetuneLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, M, vgg16, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.M = M
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.vgg16 = vgg16
        clip_model, _ = clip.load("ViT-B/32", device=device)  # Load CLIP model here
        self.clip_model = clip_model.eval()
        
        
    def custom_reshape(self, img, mode='area'):   # more to be implemented here
        reshaped_img = F.interpolate(img, (224, 224), mode=mode)
        return  reshaped_img


    def clip_preprocess(self):
        return T.Compose([
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])    
    
    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws
    
    def run_D(self, img, c, sync, fts=None):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits
    
    def run_M(self, z, c, w, sync):
        with misc.ddp_sync(self.M, sync):
            w = self.M(z=z, c=c, w=w)
        return w
    
    def accumulate_gradients(self, phase, real_images, real_c, sync, gain, img_fts, txt_fts, gen_z=None, gen_c=None,):
        z =  torch.randn(size=(img_fts, 512)).to(self.device)
        rand_img, rand_w = self.run_G(z, real_c, sync)

        lam_1 = 0.0
        lam_2 = 0.
        lam_3 = 10.
        lam_4 = 0.
        
#         def mix(w_1, w_2):
#             w_2 = w_2#*0.5 + 
#             return w_2
            
#         w_from_clip_img = self.run_M(z=real_features[0], c=real_c, sync=sync)#.repeat([1, self.G_synthesis.num_ws, 1])  # repeat the style
#         with misc.ddp_sync(self.G_synthesis, sync):
#             imgs = self.G_synthesis(w_from_clip_img)
# #         img_logits = self.run_D(imgs, real_c, sync=sync)
#         loss = lam_1*(torch.clamp(imgs, -1., 1.) - real_images).abs().sum((-1,-2,-3)).mean()
        aug_level_1 = 0.2
        random_noise = torch.randn(txt_fts.shape).to(txt_fts.device)
        random_noise = random_noise/random_noise.norm(dim=-1, keepdim=True)
        txt_fts = txt_fts*np.sqrt(1-aug_level_1) + random_noise*np.sqrt(aug_level_1)
        
#         fake_fts = torch.ones(real_features[0].size()).uniform_(-1., 1.).to(self.device)
        fake_fts = txt_fts/txt_fts.norm(dim=-1, keepdim=True)
        fake_w = self.run_M(z=fake_fts, c=real_c, w=rand_w, sync=sync)
        with misc.ddp_sync(self.G_synthesis, sync):
#             with torch.autograd.profiler.record_function('style_mixing'):
#                     cutoff = torch.empty([], dtype=torch.int64, device=fake_w.device).random_(1, fake_w.shape[1])
#                     cutoff = torch.where(torch.rand([], device=fake_w.device) < 0.5, cutoff, torch.full_like(cutoff, fake_w.shape[1]))
#                     fake_w[:, cutoff:] = rand_w[:, cutoff:]
            fake_img = self.G_synthesis(fake_w)
        normed_fake_img = self.clip_preprocess()(self.custom_reshape(fake_img))
        fake_fts_recon = self.clip_model.encode_image(normed_fake_img)
        fake_fts_recon = fake_fts_recon/fake_fts_recon.norm(dim=-1, keepdim=True)
        
#         normed_rand_img = self.clip_preprocess()(self.custom_reshape(rand_img))
#         rand_fts = self.clip_model.encode_image(normed_rand_img)
#         rand_fts = rand_fts/rand_fts.norm(dim=-1, keepdim=True)
        
#         delta_fts = fake_fts_recon - rand_fts
        loss = -lam_3*(torch.cosine_similarity(fake_fts_recon, fake_fts, dim=-1)).mean()
        
#         target_images = (real_images + 1) * (255/2)
#         if target_images.shape[2] > 256:
#             target_images = F.interpolate(target_images, size=(256, 256), mode='area')
#         target_features = self.vgg16(target_images, resize_images=False, return_lpips=True)

#         gen_images = (fake_img + 1) * (255/2)
#         if gen_images.shape[2] > 256:
#             gen_images = F.interpolate(gen_images, size=(256, 256), mode='area')
#         gen_features = self.vgg16(gen_images, resize_images=False, return_lpips=True)
#         print(gen_features.size(), target_features.size())
#         loss += lam_2*(gen_features - target_features).pow(2).sum((-1)).mean()


        
#         w_from_clip_txt = self.run_M(z=real_features[1], c=real_c, sync=sync)#.repeat([1, self.G_synthesis.num_ws, 1])  # repeat the style
#         imgs = self.G_synthesis(mix(w_sample, w_from_clip))
#         gen_images = (imgs + 1) * (255/2)
#         if gen_images.shape[2] > 256:
#             gen_images = F.interpolate(gen_images, size=(256, 256), mode='area')
#         gen_features = self.vgg16(gen_images, resize_images=False, return_lpips=True)
#         loss += lam_3*(w_from_clip_txt - w_from_clip_img).pow(2).sum(-1).mean()
        
        
#         reshaped_img = self.custom_reshape(rand_img)
#         normed_img = self.clip_preprocess()(reshaped_img)
# #         loss -= lam_4*torch.cosine_similarity(real_features[1], self.clip_model.encode_image(normed_img), dim=-1).mean()
# #         loss -= lam_4*torch.cosine_similarity(real_features[0], self.clip_model.encode_image(normed_img), dim=-1).mean()
#         recon_w = self.run_M(z=self.clip_model.encode_image(normed_img), c=real_c, sync=sync)
#         loss += lam_4*(rand_w - recon_w).pow(2).sum(-1).mean()

        training_stats.report('Loss/M/loss', loss)



        with torch.autograd.profiler.record_function('Mmain_backward'):
            loss.mean().mul(gain).backward()

#----------------------------------------------------------------------------
