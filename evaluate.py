import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.amp import autocast
from model import EfficientVSR

class EvalConfig:
    VIMEO_ROOT    = r"C:\Users\soham\OneDrive\Documents\Dataset\vimeo_triplet"
    CHECKPOINT    = './checkpoints/best_model.pth'
    OUTPUT_DIR    = './evaluation_results'
    BASE_CHANNELS = 128
    NUM_SAMPLES   = 20      
    USE_AMP       = True


def load_model(config, device):
    model = EfficientVSR(base_channels=config.BASE_CHANNELS).to(device)

    ckpt = torch.load(config.CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(best PSNR: {ckpt['best_psnr']:.2f}dB)")
    return model


def load_triplet(seq_dir):

    f1 = Image.open(os.path.join(seq_dir, 'im1.png')).convert('RGB')
    f2 = Image.open(os.path.join(seq_dir, 'im2.png')).convert('RGB')
    f3 = Image.open(os.path.join(seq_dir, 'im3.png')).convert('RGB')
    return f1, f2, f3


def prepare_inputs(f1, f2, f3, device):
    
    w, h = f1.size

    lr1 = f1.resize((w // 2, h // 2), Image.BICUBIC)
    lr3 = f3.resize((w // 2, h // 2), Image.BICUBIC)
    hr_gt = f2  


    lr1_t  = TF.to_tensor(lr1).unsqueeze(0).to(device)   
    lr3_t  = TF.to_tensor(lr3).unsqueeze(0).to(device)   
    hr_gt_t = TF.to_tensor(hr_gt).to(device)           

    return lr1_t, lr3_t, hr_gt_t, lr1, lr3, hr_gt


def bicubic_baseline(lr1_pil, lr3_pil):
 
    w, h = lr1_pil.size
    target_w, target_h = w * 2, h * 2

    lr1_arr = np.array(lr1_pil).astype(np.float32)
    lr3_arr = np.array(lr3_pil).astype(np.float32)

    avg = ((lr1_arr + lr3_arr) / 2.0).astype(np.uint8)
    avg_pil = Image.fromarray(avg)

    bicubic = avg_pil.resize((target_w, target_h), Image.BICUBIC)
    return bicubic


def compute_metrics_np(pred_np, target_np):
   
    p = psnr(target_np, pred_np, data_range=1.0)
    s = ssim(target_np, pred_np, data_range=1.0, channel_axis=2)
    return p, s

def save_comparison(sample_idx, lr1_pil, lr3_pil,
                    bicubic_pil, pred_np, gt_np,
                    model_psnr, model_ssim,
                    bicubic_psnr, bicubic_ssim,
                    output_dir):
  
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(f'Sample {sample_idx+1} — Visual Comparison',
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05)

    titles = [
        'LR Frame 1\n(Input)',
        'LR Frame 3\n(Input)',
        f'Bicubic Baseline\nPSNR: {bicubic_psnr:.2f}dB  SSIM: {bicubic_ssim:.4f}',
        f'Our Model\nPSNR: {model_psnr:.2f}dB  SSIM: {model_ssim:.4f}',
        'Ground Truth\n(Target)'
    ]

    images = [
        np.array(lr1_pil),
        np.array(lr3_pil),
        np.array(bicubic_pil),
        (pred_np * 255).clip(0, 255).astype(np.uint8),
        (gt_np   * 255).clip(0, 255).astype(np.uint8)
    ]

    for i, (title, img) in enumerate(zip(titles, images)):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    plt.savefig(
        os.path.join(output_dir, f'sample_{sample_idx+1:03d}.png'),
        dpi=120, bbox_inches='tight'
    )
    plt.close()


def save_zoomed_comparison(sample_idx, bicubic_np,
                           pred_np, gt_np, output_dir):
    
    h, w = gt_np.shape[:2]
    cy, cx = h // 2, w // 2
    crop_size = 64

    def crop(img):
        return img[cy-crop_size//2 : cy+crop_size//2,
                   cx-crop_size//2 : cx+crop_size//2]

    bicubic_crop = crop(bicubic_np)
    pred_crop    = crop(pred_np)
    gt_crop      = crop(gt_np)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle(f'Sample {sample_idx+1} — Zoomed Center Patch (64×64)',
                 fontsize=11, fontweight='bold')

    for ax, img, title in zip(axes,
        [bicubic_crop, pred_crop, gt_crop],
        ['Bicubic', 'Our Model', 'Ground Truth']):
        ax.imshow((img * 255).clip(0, 255).astype(np.uint8),
                  interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'zoomed_{sample_idx+1:03d}.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()



def plot_training_curves(log_path, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    epochs, stages = [], []
    train_psnr, val_psnr = [], []
    train_ssim, val_ssim = [], []
    train_loss = []
    char_loss, perc_loss, edge_loss, temp_loss = [], [], [], []

    with open(log_path, 'r') as f:
        next(f)  
        for line in f:
            parts = line.strip().split(',')
            epochs.append(int(parts[0]))
            stages.append(int(parts[1]))
            train_loss.append(float(parts[2]))
            train_psnr.append(float(parts[3]))
            train_ssim.append(float(parts[4]))
            val_psnr.append(float(parts[6]))
            val_ssim.append(float(parts[7]))
            char_loss.append(float(parts[8]))
            perc_loss.append(float(parts[9]))
            edge_loss.append(float(parts[10]))
            temp_loss.append(float(parts[11]))

    
    stage_changes = [i for i in range(1, len(stages))
                     if stages[i] != stages[i-1]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('EfficientVSR Training Curves', fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(epochs, train_psnr, 'b-',  label='Train PSNR', linewidth=2)
    ax.plot(epochs, val_psnr,   'r--', label='Val PSNR',   linewidth=2)
    for sc in stage_changes:
        ax.axvline(x=epochs[sc], color='gray',
                   linestyle=':', alpha=0.7)
        ax.text(epochs[sc]+0.3, min(train_psnr)*0.99,
                f'Stage {stages[sc]}', fontsize=8, color='gray')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, train_ssim, 'b-',  label='Train SSIM', linewidth=2)
    ax.plot(epochs, val_ssim,   'r--', label='Val SSIM',   linewidth=2)
    for sc in stage_changes:
        ax.axvline(x=epochs[sc], color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SSIM')
    ax.set_title('SSIM Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, char_loss, label='Charbonnier', linewidth=2)
    ax.plot(epochs, perc_loss, label='Perceptual',  linewidth=2)
    ax.plot(epochs, edge_loss, label='Edge',        linewidth=2)
    ax.plot(epochs, temp_loss, label='Temporal',    linewidth=2)
    for sc in stage_changes:
        ax.axvline(x=epochs[sc], color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Breakdown')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Training curves saved")




def main():
    config = EvalConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  EfficientVSR Evaluation")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {config.CHECKPOINT}")
    print(f"  Samples    : {config.NUM_SAMPLES}")
    print(f"{'='*60}\n")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    model = load_model(config, device)

    list_file = os.path.join(config.VIMEO_ROOT, 'tri_testlist.txt')
    with open(list_file, 'r') as f:
        test_sequences = [l.strip() for l in f if l.strip()]

    test_sequences = test_sequences[:config.NUM_SAMPLES]
    print(f"Evaluating {len(test_sequences)} sequences...\n")

    model_psnrs,    model_ssims    = [], []
    bicubic_psnrs,  bicubic_ssims  = [], []

    for idx, seq in enumerate(test_sequences):

        seq_dir = os.path.join(config.VIMEO_ROOT, 'sequences', seq)

        f1, f2, f3 = load_triplet(seq_dir)

        lr1_t, lr3_t, hr_gt_t, lr1_pil, lr3_pil, hr_gt_pil = \
            prepare_inputs(f1, f2, f3, device)

        with torch.no_grad():
            with autocast(device_type='cuda', enabled=config.USE_AMP):
                pred_t = model(lr1_t, lr3_t)  

        pred_np = pred_t.squeeze(0).cpu().float().numpy()
        pred_np = pred_np.transpose(1, 2, 0).clip(0, 1)

        gt_np = hr_gt_t.cpu().float().numpy()
        gt_np = gt_np.transpose(1, 2, 0).clip(0, 1)

        bicubic_pil = bicubic_baseline(lr1_pil, lr3_pil)
        bicubic_np  = np.array(bicubic_pil).astype(np.float32) / 255.0

        m_psnr, m_ssim = compute_metrics_np(pred_np,    gt_np)
        b_psnr, b_ssim = compute_metrics_np(bicubic_np, gt_np)

        model_psnrs.append(m_psnr)
        model_ssims.append(m_ssim)
        bicubic_psnrs.append(b_psnr)
        bicubic_ssims.append(b_ssim)

        print(f"  [{idx+1:02d}/{len(test_sequences)}] {seq}")
        print(f"         Model  → PSNR: {m_psnr:.2f}dB  SSIM: {m_ssim:.4f}")
        print(f"         Bicubic→ PSNR: {b_psnr:.2f}dB  SSIM: {b_ssim:.4f}")

        save_comparison(
            idx, lr1_pil, lr3_pil,
            bicubic_pil, pred_np, gt_np,
            m_psnr, m_ssim,
            b_psnr, b_ssim,
            config.OUTPUT_DIR
        )

        save_zoomed_comparison(
            idx, bicubic_np, pred_np, gt_np,
            config.OUTPUT_DIR
        )


    avg_model_psnr   = np.mean(model_psnrs)
    avg_model_ssim   = np.mean(model_ssims)
    avg_bicubic_psnr = np.mean(bicubic_psnrs)
    avg_bicubic_ssim = np.mean(bicubic_ssims)

    print(f"\n{'='*60}")
    print(f"  FINAL EVALUATION RESULTS ({len(test_sequences)} samples)")
    print(f"{'='*60}")
    print(f"  Method      PSNR       SSIM")
    print(f"  ─────────────────────────────")
    print(f"  Bicubic     {avg_bicubic_psnr:.2f}dB   {avg_bicubic_ssim:.4f}")
    print(f"  Our Model   {avg_model_psnr:.2f}dB   {avg_model_ssim:.4f}")
    print(f"  ─────────────────────────────")
    print(f"  Improvement {avg_model_psnr-avg_bicubic_psnr:+.2f}dB  "
          f"{avg_model_ssim-avg_bicubic_ssim:+.4f}")
    print(f"{'='*60}")

    summary_path = os.path.join(config.OUTPUT_DIR, 'results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("EfficientVSR Evaluation Results\n")
        f.write("="*40 + "\n")
        f.write(f"Samples evaluated: {len(test_sequences)}\n\n")
        f.write(f"{'Method':<15} {'PSNR':>10} {'SSIM':>10}\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Bicubic':<15} {avg_bicubic_psnr:>10.2f} "
                f"{avg_bicubic_ssim:>10.4f}\n")
        f.write(f"{'Our Model':<15} {avg_model_psnr:>10.2f} "
                f"{avg_model_ssim:>10.4f}\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Improvement':<15} {avg_model_psnr-avg_bicubic_psnr:>+10.2f} "
                f"{avg_model_ssim-avg_bicubic_ssim:>+10.4f}\n\n")
        f.write("Per-sample results:\n")
        f.write(f"{'Seq':<30} {'Model PSNR':>12} {'Bicubic PSNR':>14}\n")
        f.write("-"*60 + "\n")
        for seq, mp, bp in zip(test_sequences, model_psnrs, bicubic_psnrs):
            f.write(f"{seq:<30} {mp:>12.2f} {bp:>14.2f}\n")

    print(f"\n✅ Results summary saved: {summary_path}")

    log_path = './logs/training_log.txt'
    if os.path.exists(log_path):
        plot_training_curves(log_path, config.OUTPUT_DIR)

    print(f"\n✅ All outputs saved to: {config.OUTPUT_DIR}/")
    print(f"   - sample_XXX.png    → side by side comparisons")
    print(f"   - zoomed_XXX.png    → zoomed patch comparisons")
    print(f"   - training_curves.png → PSNR/SSIM/Loss plots")
    print(f"   - results_summary.txt → final numbers\n")


if __name__ == '__main__':
    main()