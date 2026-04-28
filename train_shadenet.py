import torch 
from models import ShadeNet
from utils import train_shadenet
from dataset_generators import Generator
from torch.utils.data import DataLoader
from torchvision import transforms as T
import yaml
import os

PATH = './config/config.yaml'


def seed_worker(worker_id):
    import random
    import numpy as np
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    

if __name__ == "__main__":
    cfg = load_config(PATH)
        

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
    

    traindataset = Generator(**cfg['traingenerator'], transform_img=transform_img, transform_mask=transform_mask)
    valdataset = Generator(**cfg['valgenerator'], transform_img=transform_img, transform_mask=transform_mask)

    valdataset = torch.utils.data.Subset(valdataset, range(0, int(len(valdataset) * 0.1)))

    trainLoader = DataLoader(traindataset, shuffle = True, **cfg['dataloader'])
    valLoader = DataLoader(valdataset, shuffle = False, **cfg['dataloader'])


    model = ShadeNet(**cfg["model"])

    model, train_diceloss, train_bceloss, train_mseloss, \
    val_diceloss,   val_bceloss,   val_mseloss = train_shadenet(model=model, train_loader=trainLoader, val_loader=valLoader, config_path=PATH, **cfg['training'])

    model = model.half()
    
    # ── Save fp16 export (inference only, do NOT train with this) ──────────
    model_fp16 = ShadeNet(**cfg["model"])
    
    model_fp16.load_state_dict(model.state_dict())
    model_fp16.half()   # convert weights to fp16

    fp16_path = os.path.join(trn['save_dir'], 'shadenet_fp16.pt')
    torch.save(model_fp16.state_dict(), fp16_path)
    print(f"Saved fp16 inference model: {fp16_path}")

