# upsampler

Python command line tool to upscale images locally with a deep-learning model downloaded from Hugging Face.

## Install

```bash
pip install .
```

## Usage

```bash
upsample image.png --factor 3 --output image-upsampled.png
```

Optional model override:

```bash
upsample image.png --factor 3 --output image-upsampled.png --model-id caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr
```
