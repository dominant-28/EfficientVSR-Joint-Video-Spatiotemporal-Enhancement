import gradio as gr
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.amp import autocast
import tempfile
import os
from model import EfficientVSR


CHECKPOINT = './checkpoints/best_model.pth'
BASE_CHANNELS = 128
USE_AMP       = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading model on {device}.....")
model =EfficientVSR(base_channels=BASE_CHANNELS).to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Model loaded - epoch {ckpt['epoch']}, " f"best PSNR {ckpt['best_psnr']:.2f}dB")

def make_divisible_by_8(frame):
    h, w   = frame.shape[:2]
    new_h  = (h // 8) * 8
    new_w  = (w // 8) * 8
    return frame[:new_h, :new_w]

def frame_to_tensor(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil       = Image.fromarray(frame_rgb)
    tensor    = TF.to_tensor(pil).unsqueeze(0).to(device)
    return tensor

def tensor_to_frame(tensor):
    arr = tensor.squeeze(0).cpu().float().numpy()
    arr = arr.transpose(1, 2, 0).clip(0, 1)
    arr = (arr * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def enhance_video(input_video_path,progress=gr.Progress()):
    if input_video_path is None:
        return None, "Please upload a video first"
    

    cap = cv2.VideoCapture(input_video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress(0, desc="Reading frames....")
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()

        if not ret:
            break   
        frames.append(make_divisible_by_8(frame))
    cap.release()

    if len(frames) == 0:
        return None, "Could not read any frames from video."

    actual_h, actual_w = frames[0].shape[:2]
    out_fps    = orig_fps * 2
    out_width  = actual_w * 2
    out_height = actual_h * 2    

    output_path = tempfile.mktemp(suffix='_enhanced.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_writer  = cv2.VideoWriter(
        output_path, fourcc,
        out_fps, (out_width, out_height)
    )

    total_pairs = len(frames) - 1
    processed   = 0

    for i in range(total_pairs):
        progress(i/total_pairs, desc= f"Enhancing frame {i+1}/{total_pairs}...")

        f1_bgr = frames[i]
        f3_bgr = frames[i+1]
        f1_t = frame_to_tensor(f1_bgr)
        f3_t = frame_to_tensor(f3_bgr)

        with torch.no_grad():
            with autocast(device_type=device.type, enabled=USE_AMP):
                f1_sr_t = model(f1_t, f1_t)
                mid_t   = model(f1_t, f3_t)

        out_writer.write(tensor_to_frame(f1_sr_t))
        out_writer.write(tensor_to_frame(mid_t))
        processed += 2
    last_t = frame_to_tensor(frames[-1])
    with torch.no_grad():
        with autocast(device_type=device.type,enabled=USE_AMP):
            last_sr = model(last_t,last_t)
    out_writer.write(tensor_to_frame(last_sr))
    out_writer.release()
    info = (
        f"✅ Enhancement complete!\n\n"
        f"Input  : {actual_w}×{actual_h} @ {orig_fps:.0f} fps  "
        f"({len(frames)} frames)\n"
        f"Output : {out_width}×{out_height} @ {out_fps:.0f} fps  "
        f"({processed+1} frames)\n\n"
        f"Resolution : {actual_w}×{actual_h} → {out_width}×{out_height}  (2× SR)\n"
        f"Frame rate : {orig_fps:.0f} fps → {out_fps:.0f} fps  (2× VFI)"
    )

    return output_path, info


with gr.Blocks(
    title="EfficientVSR",
    theme=gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
    )
) as demo:
    gr.Markdown("""
    # EfficientVSR
    ### Joint Video Super Resolution + Frame Interpolation
    Upload a low-resolution, low-framerate video.
    The model will **2× the resolution** and **2× the frame rate** simultaneously.
    """)

    gr.Markdown("""
    > **Model:** Hybrid CNN-Transformer · **Params:** 2.8M · 
    **PSNR:** +6.13 dB over bicubic · **SSIM:** 0.9088
    """)

    with gr.Row():

        with gr.Column(scale=1):
            gr.Markdown("### Input")

            input_video = gr.Video(
                label="Upload video",
                sources=["upload"],
            )
            enhance_btn = gr.Button(
                "Enhance Video",
                variant="primary",
                size="lg"
            )

            gr.Markdown("""
            **Tips for best results:**
            - Use 240p or 360p input video
            - Short clips (3-10 sec) work best
            - Video with moderate motion gives sharpest results
            """)

        with gr.Column(scale=1):
            gr.Markdown("### Output")

            output_video = gr.Video(
                label="Enhanced video (2× res, 2× fps)",
                interactive=False
            )

            info_box = gr.Textbox(
                label="Enhancement details",
                lines=6,
                interactive=False,
                placeholder="Enhancement details will appear here after processing..."
            )

    gr.Markdown("---")
    gr.Markdown("### How it works")

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            **Step 1 — Shared Encoder**  
            DenseBlock extracts rich features from both input frames
            """)
        with gr.Column():
            gr.Markdown("""
            **Step 2 — Dual Branch**  
            Temporal branch (deformable conv) + Spatial branch (SE attention) run in parallel
            """)
        with gr.Column():
            gr.Markdown("""
            **Step 3 — Cross-Frame Attention**  
            Branches communicate via windowed attention with RoPE encoding
            """)
        with gr.Column():
            gr.Markdown("""
            **Step 4 — PixelShuffle**  
            CBAM refinement + PixelShuffle upsampler outputs 2× resolution frame
            """)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            **Benchmark Results (Vimeo-90K test set)**
            | Method | PSNR | SSIM |
            |--------|------|------|
            | Bicubic baseline | 25.13 dB | 0.7509 |
            | **Our model** | **31.25 dB** | **0.9088** |
            | **Improvement** | **+6.13 dB** | **+0.1579** |
            """)
        with gr.Column():
            gr.Markdown("""
            **Architecture highlights**
            - Joint SR + VFI in single forward pass
            - RoPE positional encoding (from LLaMA)
            - Gated FFN (from LLaMA GLU)
            - Windowed cross-frame attention (from Swin)
            - Trained in 3 stages on Vimeo-90K
            """)


    enhance_btn.click(
        fn=enhance_video,
        inputs=[input_video],
        outputs=[output_video, info_box]
    )

if __name__ == '__main__':
    demo.launch(
        server_name="0.0.0.0",   
        server_port=7860,
        share=False,             
        inbrowser=True           
    )