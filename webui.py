#!/usr/bin/env python3
"""TurboDiffusion I2V Web UI"""

import os
import sys
import math
import tempfile
from pathlib import Path
from argparse import Namespace

# Set environment before imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gradio as gr
import torch
import numpy as np
from PIL import Image
from einops import rearrange, repeat
import torchvision.transforms.v2 as T

# Add turbodiffusion to path
sys.path.insert(0, str(Path(__file__).parent / "turbodiffusion" / "inference"))
sys.path.insert(0, str(Path(__file__).parent / "turbodiffusion"))

torch._dynamo.config.suppress_errors = True

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log
from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from modify_model import tensor_kwargs, create_model

# Global model cache
_models = None
_checkpoint_dir = Path(__file__).parent / "checkpoints"


def load_models():
    """Load models once and cache them."""
    global _models

    if _models is not None:
        return _models

    log.info("Loading high noise model...")
    args_high = Namespace(
        model="Wan2.2-A14B",
        attention_type="sagesla",
        sla_topk=0.1,
        quant_linear=True,
        default_norm=False,
    )
    high_noise_model = create_model(
        str(_checkpoint_dir / "TurboWan2.2-I2V-A14B-high-720P-quant.pth"),
        args_high
    ).cpu()
    torch.cuda.empty_cache()

    log.info("Loading low noise model...")
    args_low = Namespace(
        model="Wan2.2-A14B",
        attention_type="sagesla",
        sla_topk=0.1,
        quant_linear=True,
        default_norm=False,
    )
    low_noise_model = create_model(
        str(_checkpoint_dir / "TurboWan2.2-I2V-A14B-low-720P-quant.pth"),
        args_low
    ).cpu()
    torch.cuda.empty_cache()

    log.info("Loading VAE tokenizer...")
    tokenizer = Wan2pt1VAEInterface(vae_pth=str(_checkpoint_dir / "Wan2.1_VAE.pth"))

    _models = {
        "high_noise_model": high_noise_model,
        "low_noise_model": low_noise_model,
        "tokenizer": tokenizer,
    }

    log.success("All models loaded!")
    return _models


def generate_video(
    image: Image.Image,
    prompt: str,
    resolution: str,
    num_frames: int,
    num_steps: int,
    sigma_max: float,
    boundary: float,
    use_ode: bool,
    seed: int,
    progress=gr.Progress(),
):
    """Generate video from image."""
    if image is None:
        raise gr.Error("Please upload an image")

    if not prompt.strip():
        raise gr.Error("Please enter a prompt")

    # Random seed if 0
    if seed == 0:
        seed = torch.randint(0, 2**31, (1,)).item()
    seed = int(seed)
    num_frames = int(num_frames)
    log.info(f"Using seed: {seed}")

    progress(0, desc="Loading models...")
    models = load_models()
    tokenizer = models["tokenizer"]
    high_noise_model = models["high_noise_model"]
    low_noise_model = models["low_noise_model"]

    # Get text embedding
    progress(0.1, desc="Encoding prompt...")
    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=str(_checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
            prompts=prompt
        ).to(**tensor_kwargs)
    clear_umt5_memory()

    # Preprocess image
    progress(0.2, desc="Preprocessing image...")
    image = image.convert("RGB")

    # Get resolution
    aspect_ratio = "16:9"
    w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
    log.info(f"Resolution set to: {w}x{h}, frames: {num_frames}")

    lat_h = h // tokenizer.spatial_compression_factor
    lat_w = w // tokenizer.spatial_compression_factor
    lat_t = tokenizer.get_latent_num_frames(num_frames)

    # Resize while preserving aspect ratio, then pad
    orig_w, orig_h = image.size
    orig_aspect = orig_w / orig_h
    target_aspect = w / h

    if orig_aspect > target_aspect:
        # Image is wider - fit to width, pad height
        new_w = w
        new_h = int(w / orig_aspect)
    else:
        # Image is taller - fit to height, pad width
        new_h = h
        new_w = int(h * orig_aspect)

    # Resize preserving aspect ratio
    image_resized = image.resize((new_w, new_h), Image.LANCZOS)

    # Create padded image with black background
    image_padded = Image.new("RGB", (w, h), (0, 0, 0))
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2
    image_padded.paste(image_resized, (paste_x, paste_y))
    log.info(f"Image resized from {orig_w}x{orig_h} to {new_w}x{new_h}, padded to {w}x{h}")

    image_transforms = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_tensor = image_transforms(image_padded).unsqueeze(0).to(device=tensor_kwargs["device"], dtype=torch.float32)

    progress(0.25, desc="Encoding image...")
    with torch.no_grad():
        frames_to_encode = torch.cat(
            [image_tensor.unsqueeze(2), torch.zeros(1, 3, num_frames - 1, h, w, device=image_tensor.device)], dim=2
        )
        encoded_latents = tokenizer.encode(frames_to_encode)
        del frames_to_encode
        torch.cuda.empty_cache()

    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
    msk[:, :, 0, :, :] = 1.0

    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)

    condition = {
        "crossattn_emb": text_emb.to(**tensor_kwargs),
        "y_B_C_T_H_W": y
    }

    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(seed)

    init_noise = torch.randn(
        1,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][:num_steps - 1]

    t_steps = torch.tensor(
        [math.atan(sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )

    # Convert TrigFlow timesteps to RectifiedFlow
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1

    high_noise_model.cuda()
    net = high_noise_model
    switched = False

    progress(0.3, desc="Sampling...")
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        step_progress = 0.3 + 0.5 * ((i + 1) / total_steps)
        progress(step_progress, desc=f"Sampling step {i+1}/{total_steps}...")

        if t_cur.item() < boundary and not switched:
            high_noise_model.cpu()
            torch.cuda.empty_cache()
            low_noise_model.cuda()
            net = low_noise_model
            switched = True
            log.info("Switched to low noise model.")

        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition
            ).to(torch.float64)

            if use_ode:
                x = x - (t_cur - t_next) * v_pred
            else:
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape,
                    dtype=torch.float32,
                    device=tensor_kwargs["device"],
                    generator=generator,
                )

    samples = x.float()
    low_noise_model.cpu()
    torch.cuda.empty_cache()

    progress(0.85, desc="Decoding video...")
    with torch.no_grad():
        video = tokenizer.decode(samples)

    to_show = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0

    progress(0.95, desc="Saving video...")
    output_path = tempfile.mktemp(suffix=".mp4")
    save_image_or_video(rearrange(to_show, "b c t h w -> c t h (b w)"), output_path, fps=16)

    progress(1.0, desc="Done!")
    log.success(f"Video saved to {output_path}")

    return output_path


def create_ui():
    """Create Gradio UI."""
    with gr.Blocks(title="TurboDiffusion I2V", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# TurboDiffusion I2V Web UI")
        gr.Markdown("Image-to-Video generation powered by TurboDiffusion on RTX 5090")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=400,
                )
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the motion you want...",
                    lines=3,
                )

                with gr.Accordion("Advanced Settings", open=False):
                    resolution = gr.Radio(
                        choices=["480p", "720p"],
                        value="480p",
                        label="Resolution (720p may OOM)",
                    )
                    num_frames = gr.Slider(
                        minimum=17,
                        maximum=161,
                        value=81,
                        step=16,
                        label="Video Length (frames, ~16fps)",
                        info="17=1s, 49=3s, 81=5s, 113=7s, 145=9s, 161=10s (longer may OOM)",
                    )
                    num_steps = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=4,
                        step=1,
                        label="Sampling Steps",
                    )
                    sigma_max = gr.Slider(
                        minimum=80,
                        maximum=1600,
                        value=200,
                        step=10,
                        label="Sigma Max (higher = less diversity)",
                    )
                    boundary = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Model Switch Boundary",
                    )
                    use_ode = gr.Checkbox(
                        value=True,
                        label="Use ODE (sharper but less robust)",
                    )
                    seed = gr.Number(
                        value=0,
                        label="Seed",
                        precision=0,
                    )

                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_video = gr.Video(
                    label="Generated Video",
                    height=400,
                )

        generate_btn.click(
            fn=generate_video,
            inputs=[
                input_image,
                prompt,
                resolution,
                num_frames,
                num_steps,
                sigma_max,
                boundary,
                use_ode,
                seed,
            ],
            outputs=output_video,
        )

        gr.Markdown("---")
        gr.Markdown("Tips: 480p is recommended for RTX 5090. 720p may cause OOM during VAE decode.")

    return demo


if __name__ == "__main__":
    # Preload models
    print("Preloading models...")
    load_models()
    print("Starting web UI...")

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
