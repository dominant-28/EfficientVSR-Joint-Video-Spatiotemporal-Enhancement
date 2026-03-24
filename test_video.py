import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.amp import autocast
from model import EfficientVSR


class VideoConfig:
    CHECKPOINT    = './checkpoints/best_model.pth'
    BASE_CHANNELS = 128
    USE_AMP       = True

    INPUT_VIDEO   = r"C:\Users\soham\Downloads\Untitled design.mp4"
    OUTPUT_VIDEO  = './evaluation_results/output_enhanced.mp4'

    MAX_FRAMES    = None    # None for full video


def load_model(config, device):
    model = EfficientVSR(base_channels=config.BASE_CHANNELS).to(device)
    ckpt  = torch.load(config.CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"✅ Model loaded from epoch {ckpt['epoch']} "
          f"(PSNR: {ckpt['best_psnr']:.2f}dB)")
    return model


def frame_to_tensor(frame_bgr, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil       = Image.fromarray(frame_rgb)
    tensor    = TF.to_tensor(pil).unsqueeze(0).to(device)
    return tensor


def tensor_to_frame(tensor):
    arr = tensor.squeeze(0).cpu().float().numpy()
    arr = arr.transpose(1, 2, 0).clip(0, 1)
    arr = (arr * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def process_video(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  EfficientVSR Video Enhancement")
    print(f"  Device : {device}")
    print(f"  Input  : {config.INPUT_VIDEO}")
    print(f"  Output : {config.OUTPUT_VIDEO}")
    print(f"{'='*60}\n")

    model = load_model(config, device)

    cap = cv2.VideoCapture(config.INPUT_VIDEO)
    if not cap.isOpened():
        print(f"❌ Could not open: {config.INPUT_VIDEO}")
        return

    orig_fps     = cap.get(cv2.CAP_PROP_FPS)
    orig_width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if config.MAX_FRAMES:
        total_frames = min(total_frames, config.MAX_FRAMES)

   
    out_fps    = orig_fps * 2
    out_width  = orig_width  * 2
    out_height = orig_height * 2

    print(f"Input  resolution : {orig_width}×{orig_height}")
    print(f"Output resolution : {out_width}×{out_height}  (2x SR) ✅")
    print(f"Input  FPS        : {orig_fps:.1f}")
    print(f"Output FPS        : {out_fps:.1f}  (2x VFI) ✅")
    print(f"Frames to process : {total_frames}\n")

    print("Reading frames...")
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"✅ Read {len(frames)} frames\n")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(
        config.OUTPUT_VIDEO, fourcc,
        out_fps, (out_width, out_height)
    )

    print("Processing frames...")
    processed = 0

    for i in range(0, len(frames) - 1, 1):

       f1_bgr = frames[i]
       f3_bgr = frames[i + 1]  

       f1_t = frame_to_tensor(f1_bgr, device)
       f3_t = frame_to_tensor(f3_bgr, device)

   
       with torch.no_grad():
         with autocast(device_type='cuda', enabled=config.USE_AMP):
            f1_sr_t = model(f1_t, f1_t)

  
       with torch.no_grad():
         with autocast(device_type='cuda', enabled=config.USE_AMP):
            mid_t = model(f1_t, f3_t)

   
       out.write(tensor_to_frame(f1_sr_t))   
       out.write(tensor_to_frame(mid_t))   

       processed += 2

       if (i + 1) % 10 == 0:
         print(f"  Processed {i+1}/{len(frames)-1} input frames "
              f"→ {processed} output frames written")

    last_t = frame_to_tensor(frames[-1], device)
    with torch.no_grad():
       with autocast(device_type='cuda', enabled=config.USE_AMP):
          last_sr = model(last_t, last_t)
    out.write(tensor_to_frame(last_sr))
    processed += 1

    out.release()

    print(f"\n{'='*60}")
    print(f"  ✅ Enhancement Complete!")
    print(f"  Input  : {len(frames)} frames @ "
          f"{orig_width}×{orig_height} {orig_fps:.1f}fps")
    print(f"  Output : {processed} frames @ "
          f"{out_width}×{out_height} {out_fps:.1f}fps")
    print(f"  Saved  : {config.OUTPUT_VIDEO}")
    print(f"{'='*60}\n")


def make_comparison_video(config):

    print("Creating comparison video...")

    cap_orig = cv2.VideoCapture(config.INPUT_VIDEO)
    cap_enh  = cv2.VideoCapture(config.OUTPUT_VIDEO)

    enh_width  = int(cap_enh.get(cv2.CAP_PROP_FRAME_WIDTH))
    enh_height = int(cap_enh.get(cv2.CAP_PROP_FRAME_HEIGHT))
    enh_fps    = cap_enh.get(cv2.CAP_PROP_FPS)

    comp_path = config.OUTPUT_VIDEO.replace('.mp4', '_comparison.mp4')
    fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
    out       = cv2.VideoWriter(
        comp_path, fourcc,
        enh_fps, (enh_width * 2, enh_height)
    )

    frame_count = 0
    orig_frame  = None

    while True:
        ret_enh, enh_frame = cap_enh.read()
        if not ret_enh:
            break

        if frame_count % 2 == 0:
            ret_orig, orig_frame = cap_orig.read()
            if not ret_orig:
                break

            orig_up = cv2.resize(
                orig_frame,
                (enh_width, enh_height),
                interpolation=cv2.INTER_CUBIC
            )

        cv2.putText(orig_up,   'Original + Bicubic Upsample',
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
        cv2.putText(enh_frame, 'Our Model (SR + VFI)',
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        out.write(np.hstack([orig_up, enh_frame]))
        frame_count += 1

    cap_orig.release()
    cap_enh.release()
    out.release()

    print(f"✅ Comparison video saved: {comp_path}\n")


if __name__ == '__main__':
    config = VideoConfig()
    process_video(config)
    make_comparison_video(config)


