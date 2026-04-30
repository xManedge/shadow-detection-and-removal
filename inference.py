import torch
from models import ShadeNet
from torch.utils.data import DataLoader
from torchvision import transforms as T
import yaml
from dataset_generators import Generator
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

PATH = './config/config.yaml'

def load_config(path: str):
    with open(path, 'r') as f: 
        return yaml.safe_load(f)


cfg = load_config(PATH)

if cfg['inference']['useHalf']:
    model = ShadeNet(**cfg['model'])
    model.load_state_dict(torch.load(cfg['inference']['model_half_path'], map_location='cuda'))
    model.cuda()
    model.eval()
    context = torch.autocast(device_type = 'cuda', dtype=torch.float16)

else: 
    model = ShadeNet(**cfg['model'])
    model.load_state_dict(torch.load(cfg['inference']['model_path'], map_location='cuda'))
    model.cuda()
    model.eval()
    context = torch.no_grad()


t = cfg['transforms']

transform_img = T.Compose([
    T.Resize(t['img_resize']),
    T.ToTensor(),
    T.Normalize(mean=t['img_mean'], std=t['img_std']),
    ])

transform_mask = T.Compose([
    T.Resize(t['mask_resize'], interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor(),
    ])

transform_target = T.Compose([
    T.Resize(t['img_resize']),
    T.ToTensor(),
])

dataloader_config = dict(cfg['dataloader'])
dataloader_config['batch_size'] = 64


testSet = Generator(**cfg['valgenerator'], transform_img=transform_img, transform_mask=transform_mask, transform_target=transform_target)
testLoader = DataLoader(testSet, shuffle=False, **dataloader_config)
testLoader_tqdm = tqdm(testLoader, total=len(testLoader))

# testing 
os.makedirs(cfg['inference']['save_dir'], exist_ok=True)
for batch_idx, (img, _, _) in enumerate(testLoader_tqdm):
    img = img.cuda()
    if cfg['inference']['useHalf']: 
        img = img.half()
    with torch.no_grad(), context:
        mask, reconstructed = model(img)
    
     # iterate over batch
    for i in range(img.shape[0]):
        sample_idx = batch_idx * dataloader_config['batch_size'] + i

        # save mask
        mask_np = mask[i].squeeze(0).cpu().float().clamp(0, 1)
        mask_np = (mask_np.numpy() * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_np, mode='L')
        mask_img.save(os.path.join(cfg['inference']['save_dir'], f'{sample_idx}_mask.png'))

        # save reconstructed
        recon_np = reconstructed[i].cpu().float().clamp(0, 1)
        recon_np = (recon_np.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        recon_img = Image.fromarray(recon_np, mode='RGB')
        recon_img.save(os.path.join(cfg['inference']['save_dir'], f'{sample_idx}_reconstructed.png'))




