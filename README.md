# TurboDiffusion WebUI for RTX 5090

RTX 5090向けのImage-to-Video生成WebUIです。[TurboDiffusion](https://github.com/thu-ml/TurboDiffusion)をベースに、Gradio WebUIを追加しています。

## 必要環境

- NVIDIA RTX 5090 (32GB VRAM)
- Python 3.12
- CUDA 12.x
- uv (推奨) または pip

## クイックスタート

### 1. リポジトリのクローン

```bash
git clone https://github.com/shi3z/TurboDiffusion-5090-webui.git
cd TurboDiffusion-5090-webui
git submodule update --init --recursive
```

### 2. 環境構築 (uv使用)

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch>=2.7.0 ninja
uv pip install -e . --no-build-isolation
uv pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation
uv pip install gradio
```

### 3. チェックポイントのダウンロード

```bash
mkdir -p checkpoints
cd checkpoints

# VAE & Text Encoder
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth

# I2V Models (量子化版、RTX 5090用)
wget https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-high-720P-quant.pth
wget https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-low-720P-quant.pth

cd ..
```

### 4. WebUI起動

```bash
source .venv/bin/activate
export PYTHONPATH=turbodiffusion
python webui.py
```

起動後、以下のURLでアクセス:
- ローカル: http://localhost:7860
- LAN/VPN: http://<サーバーIP>:7860

## WebUI機能

| パラメータ | 説明 |
|-----------|------|
| Resolution | 480p (推奨) / 720p (OOMの可能性あり) |
| Video Length | 17-161フレーム (1-10秒) |
| Sampling Steps | 1-4 (多いほど高品質) |
| Sigma Max | 80-1600 (高いほど多様性低下) |
| Seed | 0=ランダム、それ以外は固定 |
| Use ODE | ON=シャープ、OFF=ロバスト |

## 注意事項

- 480p推奨。720pはVAEデコード時にOOMになる場合があります
- 長い動画 (113フレーム以上) は480pで使用してください
- 入力画像は縦横比を維持してパディングされます

## 元リポジトリ

- [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion)
- [Paper](https://arxiv.org/pdf/2512.16093)

## License

Apache License 2.0
