import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torchvision.models as models
from torchvision.models import VGG16_Weights
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from dataset import get_dataloaders
from model import EfficientVSR

class Config:
    VIMEO_ROOT = r"C:\Users\soham\OneDrive\Documents\Dataset\vimeo_triplet"
    SAVE_DIR = './checkpoints'
    LOG_DIR = './logs'

    PATCH_SIZE = 64
    MAX_TRAIN_SAMPLES = 20000
    MAX_TEST_SAMPLES = 2000
    NUM_WORKERS = 1

    BASE_CHANNELS = 128
    BATCH_SIZE = 8
    TOTAL_EPOCHS = 60
    INIT_LR = 1e-4
    LR_DECAY_STEP = 20
    LR_DECAY_GAMMA = 0.5
    WEIGHT_DECAY = 1e-4

    W_CHARBONNIER = 1.0
    W_PERCEPTUAL  = 0.1
    W_EDGE        = 0.05
    W_TEMPORAL    = 0.01

    STAGE1_EPOCHS = 10   
    STAGE2_EPOCHS = 10   
    STAGE3_EPOCHS = 40 

    USE_AMP = True
    SAVE_EVERY = 5


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon =epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff*diff + self.epsilon ** 2) 
        return loss.mean() 

class PerceptualLoss(nn.Module):
    def __init__(self,device):
        super().__init__()

        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor =self.feature_extractor.to(device)
        self.criterion = nn.L1Loss()

    def forward(self,pred,target):

        pred_features = self.feature_extractor(pred)
        target_features =self.feature_extractor(target)

        return self.criterion(pred_features,target_features)


class EdgeLoss(nn.Module):
    def __init__(self,device):
        super().__init__()

        sobel_x = torch.tensor([
            [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1]
        ], dtype=torch.float32)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32)
        self.register_buffer('sobel_x',
            sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y',
            sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        
        self.criterion =nn.L1Loss()
        self.to(device)

    def _get_edges(self, x):
        edge_x =torch.nn.functional.conv2d(x,self.sobel_x,padding=1,groups =3)
        edge_y =torch.nn.functional.conv2d(x,self.sobel_y,padding=1,groups=3)

        return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)


    def forward(self, pred,target):
        pred_edges =self._get_edges(pred)
        target_edges =self._get_edges(target)

        return self.criterion(pred_edges,target_edges)

class TemporalLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = CharbonnierLoss()

    def forward(self, pred, lr1, lr3):
        lr1_up =torch.nn.functional.interpolate(
            lr1, size=pred.shape[2:], mode='bilinear', align_corners=False
        )    
        lr3_up =torch.nn.functional.interpolate(
            lr3, size= pred.shape[2:], mode='bilinear',align_corners=False
        )
        temporal_ref = (lr1_up + lr3_up) / 2.0
        return self.criterion(pred,temporal_ref)
    
class TotalLoss(nn.Module):
    def __init__(self,config,device):
        super().__init__()
        self.w_char = config.W_CHARBONNIER
        self.w_perc = config.W_PERCEPTUAL
        self.w_edge = config.W_EDGE
        self.w_temp = config.W_TEMPORAL

        self.char_loss  = CharbonnierLoss()
        self.perc_loss  = PerceptualLoss(device)
        self.edge_loss  = EdgeLoss(device)
        self.temp_loss  = TemporalLoss()

    def forward(self,pred,target,lr1,lr3):
        l_char =self.char_loss(pred,target)
        l_perc =self.perc_loss(pred, target)
        l_edge =self.edge_loss(pred,target)
        l_temp =self.temp_loss(pred, lr1, lr3)

        total = (self.w_char*l_char + self.w_perc*l_perc + self.w_edge*l_edge + self.w_temp*l_temp)

        return total, {
            'charbonnier': l_char.item(),
            'perceptual':  l_perc.item(),
            'edge':        l_edge.item(),
            'temporal':    l_temp.item(),
            'total':       total.item()}    


def compute_metrics(pred, target):
    
    pred_np   = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    psnr_vals = []
    ssim_vals = []

    for i in range(pred_np.shape[0]):
        
        p = pred_np[i].transpose(1, 2, 0).clip(0, 1)
        t = target_np[i].transpose(1, 2, 0).clip(0, 1)

        psnr_vals.append(psnr(t, p, data_range=1.0))
        ssim_vals.append(ssim(t, p,
                              data_range=1.0,
                              channel_axis=2))

    return np.mean(psnr_vals), np.mean(ssim_vals)


def set_stage(model, stage):
 
    for param in model.parameters():
        param.requires_grad = False

    if stage == 1:
        print("\n[Stage 1] Training SR branch only")
        for name, param in model.named_parameters():
            if any(x in name for x in [
                'input_conv', 'encoder',          
                'se_attn', 'spatial_res',          
                'spatial_gffn',                   
                'fuse_conv', 'cbam', 'upsample'   
            ]):
                param.requires_grad = True

    elif stage == 2:
        print("\n[Stage 2] Training VFI branch only")

        for name, param in model.named_parameters():
            if any(x in name for x in [
                'input_conv', 'encoder',          
                'deform_align',                   
                'temporal_res', 'temporal_gffn',  
                'fuse_conv', 'cbam', 'upsample'    
            ]):
                param.requires_grad = True

    elif stage == 3:
        print("\n[Stage 3] Training full model jointly") 
        for param in model.parameters():
            param.requires_grad = True


    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")


def train_one_epoch(model, loader, optimizer,
                    criterion, scaler, device, epoch, config):
    model.train()

    total_loss   = 0
    loss_details = {'charbonnier': 0, 'perceptual': 0,
                    'edge': 0, 'temporal': 0, 'total': 0}
    total_psnr   = 0
    total_ssim   = 0
    num_batches  = len(loader)
    start_time   = time.time()

    for batch_idx, (lr1, lr3, hr_mid) in enumerate(loader):

        lr1    = lr1.to(device, non_blocking=True)
        lr3    = lr3.to(device, non_blocking=True)
        hr_mid = hr_mid.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=config.USE_AMP):
            pred = model(lr1, lr3)
            loss, details = criterion(pred, hr_mid, lr1, lr3)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            batch_psnr, batch_ssim = compute_metrics(pred, hr_mid)

        total_loss += details['total']
        total_psnr += batch_psnr
        total_ssim += batch_ssim

        for k in loss_details:
            loss_details[k] += details[k]

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch [{epoch}] "
                  f"Batch [{batch_idx+1}/{num_batches}] "
                  f"Loss: {details['total']:.4f} "
                  f"PSNR: {batch_psnr:.2f}dB "
                  f"SSIM: {batch_ssim:.4f} "
                  f"Time: {elapsed:.1f}s")

    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    for k in loss_details:
        loss_details[k] /= num_batches

    return avg_loss, avg_psnr, avg_ssim, loss_details



def validate(model, loader, criterion, device, config):
    model.eval()

    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = len(loader)

    with torch.no_grad():
        for lr1, lr3, hr_mid in loader:
            lr1    = lr1.to(device, non_blocking=True)
            lr3    = lr3.to(device, non_blocking=True)
            hr_mid = hr_mid.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=config.USE_AMP):
                pred = model(lr1, lr3)
                loss, details = criterion(pred, hr_mid, lr1, lr3)

            batch_psnr, batch_ssim = compute_metrics(pred, hr_mid)

            total_loss += details['total']
            total_psnr += batch_psnr
            total_ssim += batch_ssim

    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches

    return avg_loss, avg_psnr, avg_ssim

def save_checkpoint(model, optimizer, scheduler,
                    epoch, best_psnr, config, filename):
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    path = os.path.join(config.SAVE_DIR, filename)
    torch.save({
        'epoch':      epoch,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict(),
        'best_psnr':  best_psnr,
    }, path)
    print(f"  ✅ Saved checkpoint: {path}")


def load_checkpoint(model, optimizer, scheduler, config, filename):
    path = os.path.join(config.SAVE_DIR, filename)
    if not os.path.exists(path):
        print(f"  No checkpoint found at {path}")
        return 0, 0.0

    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    print(f"  ✅ Loaded checkpoint: {path}")
    return ckpt['epoch'], ckpt['best_psnr']

class Logger:
   

    def __init__(self, config):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self.path = os.path.join(config.LOG_DIR, 'training_log.txt')

        with open(self.path, 'w') as f:
            f.write("epoch,stage,train_loss,train_psnr,train_ssim,"
                    "val_loss,val_psnr,val_ssim,"
                    "char_loss,perc_loss,edge_loss,temp_loss\n")

    def log(self, epoch, stage, train_metrics,
            val_metrics, loss_details):
        train_loss, train_psnr, train_ssim = train_metrics
        val_loss,   val_psnr,   val_ssim   = val_metrics

        line = (f"{epoch},{stage},"
                f"{train_loss:.6f},{train_psnr:.4f},{train_ssim:.6f},"
                f"{val_loss:.6f},{val_psnr:.4f},{val_ssim:.6f},"
                f"{loss_details['charbonnier']:.6f},"
                f"{loss_details['perceptual']:.6f},"
                f"{loss_details['edge']:.6f},"
                f"{loss_details['temporal']:.6f}\n")

        with open(self.path, 'a') as f:
            f.write(line)

def main():

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  EfficientVSR Training")
    print(f"  Device : {device}")
    print(f"  Epochs : {config.TOTAL_EPOCHS}")
    print(f"  Batch  : {config.BATCH_SIZE}")
    print(f"  AMP    : {config.USE_AMP}")
    print(f"{'='*60}\n")

    train_loader, val_loader = get_dataloaders(
        vimeo_root        = config.VIMEO_ROOT,
        patch_size        = config.PATCH_SIZE,
        batch_size        = config.BATCH_SIZE,
        max_train_samples = config.MAX_TRAIN_SAMPLES,
        max_test_samples  = config.MAX_TEST_SAMPLES,
        num_workers       = config.NUM_WORKERS
    )

    model = EfficientVSR(base_channels=config.BASE_CHANNELS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = TotalLoss(config, device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.INIT_LR,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_DECAY_STEP,
        gamma=config.LR_DECAY_GAMMA
    )

    scaler = GradScaler('cuda', enabled=config.USE_AMP)


    logger = Logger(config)

  
    stage_schedule = {}
    for e in range(1, config.STAGE1_EPOCHS + 1):
        stage_schedule[e] = 1
    for e in range(config.STAGE1_EPOCHS + 1,
                   config.STAGE1_EPOCHS + config.STAGE2_EPOCHS + 1):
        stage_schedule[e] = 2
    for e in range(config.STAGE1_EPOCHS + config.STAGE2_EPOCHS + 1,
                   config.TOTAL_EPOCHS + 1):
        stage_schedule[e] = 3


    best_psnr    = 0.0
    current_stage = 0

    for epoch in range(1, config.TOTAL_EPOCHS + 1):

        stage = stage_schedule[epoch]
        if stage != current_stage:
            current_stage = stage
            set_stage(model, stage)


            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.INIT_LR,
                weight_decay=config.WEIGHT_DECAY
            )
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.LR_DECAY_STEP,
                gamma=config.LR_DECAY_GAMMA
            )

        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{config.TOTAL_EPOCHS} | "
              f"Stage {stage} | LR: {current_lr:.6f}")
        print(f"{'─'*60}")


        train_loss, train_psnr, train_ssim, loss_details = train_one_epoch(
            model, train_loader, optimizer,
            criterion, scaler, device, epoch, config
        )

  
        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, criterion, device, config
        )

       
        scheduler.step()

      
        print(f"\n  📊 Epoch {epoch} Summary:")
        print(f"     Train → Loss: {train_loss:.4f} | "
              f"PSNR: {train_psnr:.2f}dB | SSIM: {train_ssim:.4f}")
        print(f"     Val   → Loss: {val_loss:.4f}   | "
              f"PSNR: {val_psnr:.2f}dB | SSIM: {val_ssim:.4f}")
        print(f"     Losses → Char: {loss_details['charbonnier']:.4f} | "
              f"Perc: {loss_details['perceptual']:.4f} | "
              f"Edge: {loss_details['edge']:.4f} | "
              f"Temp: {loss_details['temporal']:.4f}")

     
        logger.log(
            epoch, stage,
            (train_loss, train_psnr, train_ssim),
            (val_loss,   val_psnr,   val_ssim),
            loss_details
        )

      
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, optimizer, scheduler,
                            epoch, best_psnr, config, 'best_model.pth')
            print(f"  🏆 New best PSNR: {best_psnr:.2f}dB")

       
        if epoch % config.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, scheduler,
                            epoch, best_psnr, config,
                            f'epoch_{epoch}.pth')

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Validation PSNR: {best_psnr:.2f}dB")
    print(f"  Checkpoints saved in: {config.SAVE_DIR}")
    print(f"  Training log saved in: {config.LOG_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
