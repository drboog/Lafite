
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

class Model(torch.nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(512, 1024)
        self.linear2 = torch.nn.Linear(1024, 1024)
        self.linear3 = torch.nn.Linear(1024, 1024)
        self.linear4 = torch.nn.Linear(1024, 512)
        self.linear5 = torch.nn.Linear(512, 1024)
        self.linear6 = torch.nn.Linear(1024, 1024)
        self.linear7 = torch.nn.Linear(1024, 1024)
        self.linear8 = torch.nn.Linear(1024, 512)
        self.device = device

    def forward(self, x):
        mu = F.leaky_relu(self.linear1(x))
        mu = F.leaky_relu(self.linear2(mu))
        mu = F.leaky_relu(self.linear3(mu))
        mu = self.linear4(mu)
        std = F.leaky_relu(self.linear5(x))
        std = F.leaky_relu(self.linear6(std))
        std = F.leaky_relu(self.linear7(std))
        std = self.linear8(std)
        return mu + std.exp()*(torch.randn(mu.shape).to(self.device))
    
    def loss(self, real, fake, temp=0.1, lam=0.5):
        sim = torch.cosine_similarity(real.unsqueeze(1), fake.unsqueeze(0), dim=-1)
        if temp > 0.:
            sim = torch.exp(sim/temp)
            sim1 = torch.diagonal(F.softmax(sim, dim=1))*temp
            sim2 = torch.diagonal(F.softmax(sim, dim=0))*temp
            if 0.<lam < 1.:
                return -(lam*torch.log(sim1) + (1.-lam)*torch.log(sim2))
            elif lam == 0:
                return -torch.log(sim2)
            else:
                return -torch.log(sim1)
        else:
            return -torch.diagonal(sim)
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
        self.mapper = Model(device)
        self.mapper.load_state_dict(torch.load('./implicit.0.001.64.True.0.0.pth', map_location='cpu')) # path to the noise mapping network
        self.mapper.to(device)
        
        
    def run_G(self, z, c, sync, txt_fts=None, ):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
                
            if self.style_mixing_prob > 0:
                new_ws = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)

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
    
    def full_preprocess(self, img, mode='bicubic', ratio=0.5):
        full_size = img.shape[-2]

        if full_size < 224:
            pad_1 = torch.randint(0, 224-full_size, ())
            pad_2 = torch.randint(0, 224-full_size, ())
            m = torch.nn.ConstantPad2d((pad_1, 224-full_size-pad_1, pad_2, 224-full_size-pad_2), 1.)
            reshaped_img = m(img)
        else:
            cut_size = torch.randint(int(ratio*full_size), full_size, ())
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
            sim = torch.exp(sim/temp) # This implementation is incorrect, it should be sim=sim/temp.
            # However, this incorrect implementation can reproduce our results with provided hyper-parameters.
            # If you want to use the correct implementation, please manually revise it.
            # The correct implementation should lead to better results, but don't use our provided hyper-parameters, you need to carefully tune lam, temp, itd, itc and other hyper-parameters
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

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, img_fts, txt_fts, lam, temp, gather, d_use_fts, itd, itc, iid, iic, mixing_prob=0.):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        
        # augmentation
        aug_level_1 = 0.1
        aug_level_2 = 0.75
#         print(torch.cosine_similarity(img_fts, txt_fts, dim=-1))

        mixing_prob = mixing_prob # probability to use img_fts instead of txt_fts
        random_noise = torch.randn(txt_fts.shape).to(img_fts.device)# + torch.randn((1, 512)).to(img_fts.device)
        random_noise = random_noise/random_noise.norm(dim=-1, keepdim=True)
        
        txt_fts_ = txt_fts*(1-aug_level_1) + random_noise*aug_level_1
        txt_fts_ = txt_fts_/txt_fts_.norm(dim=-1, keepdim=True)
        if txt_fts.shape[-1] == img_fts.shape[-1]:
# #             Gaussian purterbation
            img_fts_ = img_fts*(1-aug_level_2) + random_noise*aug_level_2

            # learned generation
#             with torch.no_grad():
#                 normed_real_full_img = self.full_preprocess(real_img, ratio=0.99)
#                 img_fts_real_full_ = self.clip_model.encode_image(normed_real_full_img).float()
#                 img_fts_real_full_ = img_fts_real_full_/img_fts_real_full_.norm(dim=-1, keepdim=True)
                
#                 # img_fts_real_full_ = img_fts
#                 img_fts_ = self.mapper(img_fts_real_full_) + img_fts_real_full_
            
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
        temp = temp
        lam = lam
        

        def gather_tensor(input_tensor, gather_or_not):
            if gather_or_not:
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                output_tensor = [torch.zeros_like(input_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(output_tensor, input_tensor)
                output_tensor[rank] = input_tensor
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


