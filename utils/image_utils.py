import os
from pathlib import Path
from PIL import Image

def preprocess_images(input_dir, output_dir, size=(512, 512)):
    """
    Simple ETL utility for raw image datasets.
    I use this to normalize varied source images into a strict 512x512 bucket 
    required for Stable Diffusion 1.5 LoRA training.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Making sure the target directory exists before dumping files.
    output_path.mkdir(parents=True, exist_ok=True)

    # Allowed formats. I included WebP because a lot of game assets come in that format these days.
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    count = 0

    print(f"[*] Starting ingestion from directory: {input_dir}")

    for file_path in input_path.iterdir():
        # Quick extension check (case-insensitive).
        if file_path.suffix.lower() in valid_exts:
            try:
                # Force convert to RGB. 
                # Dropping the Alpha channel (RGBA) is crucial because SD models usually fail with 4-channel tensors.
                img = Image.open(file_path).convert("RGB")

                # Center Crop Logic:
                # I need a 1:1 aspect ratio. Instead of squashing the image (distortion),
                # I calculate the center square and crop it.
                width, height = img.size
                min_dim = min(width, height)

                left = (width - min_dim) / 2
                top = (height - min_dim) / 2
                right = (width + min_dim) / 2
                bottom = (height + min_dim) / 2

                # Crop first, then resize.
                img = img.crop((left, top, right, bottom))
                
                # Using Lanczos filter for downsampling to maintain edge sharpness.
                # Bilinear is faster, but Lanczos is better for training data quality.
                img = img.resize(size, Image.Resampling.LANCZOS)

                # Saving as PNG to avoid compression artifacts introduced by JPEG.
                save_path = output_path / f"{file_path.stem}.png"
                img.save(save_path)
                count += 1
            except Exception as e:
                # Logging the error but not stopping the pipeline. 
                # One bad file shouldn't crash the whole batch.
                print(f"[!] Skipping corrupt file {file_path.name}: {e}")
    
    print(f"[+] Preprocessing complete. {count} images are ready in {output_dir}")

if __name__ == "__main__":
    # Standard entry point for manual execution.
    preprocess_images("data/raw", "data/processed")