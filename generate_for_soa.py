import torch
import numpy as np
import pickle
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import clip
import sys


class Generator:
    def __init__(self, device, path):
        self.name = 'generator'
        self.model = self.load_model(device, path)
        self.device = device
        self.force_32 = False
        
    def load_model(self, device, path):
        with dnnlib.util.open_url(path) as f:
            network= legacy.load_network_pkl(f)
            self.G_ema = network['G_ema'].to(device)
            self.D = network['D'].to(device)
#                 self.G = network['G'].to(device)
            return self.G_ema
        
    def generate(self, latent, c, fts, noise_mode='const', return_styles=True):
        return self.model(latent, c, fts=fts, noise_mode=noise_mode, return_styles=return_styles, force_fp32=self.force_32)
    
    def generate_from_style(self, style, noise_mode='const'):
        ws = torch.randn(style[0].shape[0], self.model.num_ws, 512)
        return self.model.synthesis(ws, fts=None, styles=style, noise_mode=noise_mode, force_fp32=self.force_32)
    
    def tensor_to_img(self, tensor):
        img = torch.clamp((tensor + 1.) * 127.5, 0., 255.)
        img_list = img.permute(0, 2, 3, 1)
        img_list = [img for img in img_list]
        return Image.fromarray(torch.cat(img_list, dim=-2).detach().cpu().numpy().astype(np.uint8))


sys.path.append("path_to_Lafite")  # revise the path here
import dnnlib, legacy
import torch.nn.functional as F
import torchvision.transforms as T

device = 'cuda:0'
path = 'pre-trained_model.pkl'  # path to the pre-trained model
direct = os.walk('./SOA/captions/')  # see https://github.com/tohinz/semantic-object-accuracy-for-generative-text-to-image-synthesis/tree/master/SOA
for a,b,c in direct:
    path = a
    file_list = c

generator = Generator(device=device, path=path)
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model = clip_model.eval()


with torch.no_grad():
    c = torch.zeros((1, generator.model.c_dim), device=device)
    for name in tqdm(file_list[:]):
        count = 0
        if '00' in name:
            img_per_cap = 1
        else:
            img_per_cap = 3
        
        img_path = './SOA/images/'+name[:8]  # will generate images
        print(img_path)
        Path(img_path).mkdir(parents=True, exist_ok=True)
        with open('./SOA/captions/' + str(name), 'rb') as f:
            captions = pickle.load(f)
            for cap in captions:
                if count <= 30000 or img_per_cap == 3:
                    txt = cap['caption'].replace('/', ' ')
                    tokenized_txt = clip.tokenize([txt]).to(device)
                    txt_fts = clip_model.encode_text(tokenized_txt)
                    txt_fts = txt_fts/txt_fts.norm(dim=-1, keepdim=True)
                    for j in range(img_per_cap):
                        z = torch.randn((1, 512)).to(device)
                        img, _ = generator.generate(latent=z, c=c, fts=txt_fts)
                        to_show_img = generator.tensor_to_img(img)
                        to_show_img.save(os.path.join(img_path, txt + str(j)+'.png'))
                        count += 1
