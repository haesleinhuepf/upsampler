from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
MAX_UPSAMPLING_PASSES = 16


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="upsample",
        description="Upsample an image with a deep learning model downloaded from Hugging Face.",
    )
    parser.add_argument("image", help="Path to the input image file.")
    parser.add_argument(
        "--factor",
        type=float,
        required=True,
        help="Upsampling factor (> 1). Example: 3",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the upsampled image.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID}).",
    )
    return parser.parse_args(argv)


def _load_model(model_id: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model, device


def _upsample_pass(image: Any, processor: Any, model: Any, device: Any) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output = model(**inputs)

    reconstruction = output.reconstruction.data.squeeze().float().cpu().clamp_(0, 1)
    data = reconstruction.permute(1, 2, 0).numpy()
    return Image.fromarray((data * 255.0).round().astype(np.uint8))


def upsample_image(input_path: Path, output_path: Path, factor: float, model_id: str) -> None:
    if factor <= 1:
        raise ValueError("factor must be greater than 1")

    if not input_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")

    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processor, model, device = _load_model(model_id)

    with Image.open(input_path) as source:
        image = source.convert("RGB")

    target_width = max(1, round(image.width * factor))
    target_height = max(1, round(image.height * factor))

    result = image
    passes = 0
    while result.width < target_width or result.height < target_height:
        previous_size = (result.width, result.height)
        result = _upsample_pass(result, processor, model, device)
        passes += 1

        if (result.width, result.height) == previous_size:
            raise RuntimeError(
                "Upsampling model did not increase image size "
                f"(current: {result.width}x{result.height}, target: {target_width}x{target_height}); "
                "cannot reach requested factor."
            )
        if passes >= MAX_UPSAMPLING_PASSES:
            raise RuntimeError(
                f"Exceeded maximum upsampling passes ({MAX_UPSAMPLING_PASSES}) "
                f"before reaching target size {target_width}x{target_height}."
            )

    if result.width != target_width or result.height != target_height:
        result = result.resize((target_width, target_height), Image.Resampling.LANCZOS)

    result.save(output_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    upsample_image(Path(args.image), Path(args.output), args.factor, args.model_id)
    return 0
