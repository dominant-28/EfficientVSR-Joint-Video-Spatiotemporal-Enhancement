import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class VimeoTripletDataset(Dataset):
    def __init__(self,
                 vimeo_root,          
                 split='train',       
                 patch_size=64,       
                 max_samples=10000,   
                 augment=True):       

        self.vimeo_root = vimeo_root
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and (split == 'train')

        list_file = os.path.join(
            vimeo_root,
            'tri_trainlist.txt' if split == 'train' else 'tri_testlist.txt'
        )

        with open(list_file, 'r') as f:
            lines = f.read().splitlines()

        self.samples = [l for l in lines if l.strip()]

        if max_samples and len(self.samples) > max_samples:
            random.seed(42)  
            self.samples = random.sample(self.samples, max_samples)

        print(f"[Dataset] {split} set: {len(self.samples)} triplets loaded")

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _downsample(self, img, scale=2):

        w, h = img.size
        small = img.resize((w // scale, h // scale), Image.BICUBIC)
        return small

    def _random_crop(self, imgs, patch_size):
        w, h = imgs[0].size
        
        if w < patch_size or h < patch_size:
            imgs = [img.resize((max(w, patch_size), max(h, patch_size)), 
                               Image.BICUBIC) for img in imgs]
            w, h = imgs[0].size

        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        return [TF.crop(img, y, x, patch_size, patch_size) for img in imgs]

    def _augment(self, imgs):

        if random.random() > 0.5:
            imgs = [TF.hflip(img) for img in imgs]

        if random.random() > 0.5:
            imgs = [TF.vflip(img) for img in imgs]

        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            imgs = [TF.rotate(img, angle) for img in imgs]

        return imgs

    def _to_tensor(self, img):
        return TF.to_tensor(img)  

    def __getitem__(self, idx):
        
        sample_path = self.samples[idx]
        seq_dir = os.path.join(self.vimeo_root, 'sequences', sample_path)

        f1 = self._load_frame(os.path.join(seq_dir, 'im1.png'))
        f2 = self._load_frame(os.path.join(seq_dir, 'im2.png'))
        f3 = self._load_frame(os.path.join(seq_dir, 'im3.png'))

        if self.split == 'train':

            f1, f2, f3 = self._random_crop([f1, f2, f3], self.patch_size)

            if self.augment:
                f1, f2, f3 = self._augment([f1, f2, f3])

        lr1 = self._downsample(f1, scale=2)
        lr3 = self._downsample(f3, scale=2)
        hr_mid = f2  

        lr1    = self._to_tensor(lr1)
        lr3    = self._to_tensor(lr3)
        hr_mid = self._to_tensor(hr_mid)

        return lr1, lr3, hr_mid



def get_dataloaders(vimeo_root,
                    patch_size=64,
                    batch_size=8,
                    max_train_samples=10000,
                    max_test_samples=1000,
                    num_workers=2):

    train_dataset = VimeoTripletDataset(
        vimeo_root=vimeo_root,
        split='train',
        patch_size=patch_size,
        max_samples=max_train_samples,
        augment=True
    )

    test_dataset = VimeoTripletDataset(
        vimeo_root=vimeo_root,
        split='test',
        patch_size=patch_size,
        max_samples=max_test_samples,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,      
        drop_last=True           
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,            
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader



if __name__ == '__main__':
    
    VIMEO_ROOT = r"C:\Users\soham\OneDrive\Documents\Dataset\vimeo_triplet"  
    
    train_loader, test_loader = get_dataloaders(
        vimeo_root=VIMEO_ROOT,
        patch_size=64,
        batch_size=8,
        max_train_samples=10000,
        max_test_samples=1000
    )

    lr1, lr3, hr_mid = next(iter(train_loader))

    print(f"\n✅ Batch loaded successfully!")
    print(f"   LR Frame 1 shape : {lr1.shape}")  
    print(f"   LR Frame 3 shape : {lr3.shape}")      
    print(f"   HR Target shape  : {hr_mid.shape}")  
    print(f"   LR value range   : [{lr1.min():.3f}, {lr1.max():.3f}]")
    print(f"   HR value range   : [{hr_mid.min():.3f}, {hr_mid.max():.3f}]")
    print(f"\n   Train batches : {len(train_loader)}")
    print(f"   Test batches  : {len(test_loader)}")