import argparse
import yaml
import os
import subprocess
import sys
import json

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Loading the project configuration to keep paths dynamic.
    config = load_config("config.yaml")
    
    MODEL_NAME = config["model"]["base_model"]
    OUTPUT_DIR = config["model"]["output_dir"]
    # I'm pointing this to the 'processed' folder where I already cropped/resized the images.
    DATA_DIR = "data/processed" 
    
    # This is the trigger token. I'm using "sks" (a rare token) + "style" 
    # to force the model to learn this specific art direction without bleeding into other concepts.
    INSTANCE_PROMPT = "sks villain style" 
    
    print(f"[*] Initializing LoRA training sequence for: {MODEL_NAME}")
    
    # Instead of writing a training loop from scratch (which is error-prone), 
    # I'm fetching the official Diffusers training script. 
    # It's robust, well-tested, and handles the backpropagation math perfectly.
    script_url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py"
    script_name = "train_text_to_image_lora.py"
    
    if not os.path.exists(script_name):
        print(f"[*] Training script not found. Fetching from HuggingFace...")
        # Assuming curl is available in the env. If not, requests library could be used here.
        subprocess.run(["curl", "-O", script_url])
    
    # --- AUTOMATIC DATASET LABELING ---
    # The training script expects a metadata file linking images to text.
    # Since I don't have a manually labeled dataset, I'm automating this step.
    # I'm tagging every image with the instance prompt to enforce the style transfer.
    metadata_path = os.path.join(DATA_DIR, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        print("[*] Generating default metadata.jsonl for training...")
        with open(metadata_path, "w") as f:
            for img in os.listdir(DATA_DIR):
                if img.endswith((".png", ".jpg", ".jpeg")):
                    entry = {"file_name": img, "text": INSTANCE_PROMPT}
                    f.write(json.dumps(entry) + "\n")
    
    # --- CONSTRUCTING THE TRAIN COMMAND ---
    # I'm using 'accelerate launch' to handle mixed precision (fp16) automatically.
    # The arguments are tuned specifically for a consumer GPU (like my RTX 3050).
    command = [
        "accelerate", "launch", script_name,
        "--pretrained_model_name_or_path=" + MODEL_NAME,
        "--train_data_dir=" + DATA_DIR,
        "--output_dir=" + OUTPUT_DIR,
        
        # Keeping resolution at 512 to match SD 1.5 native res and save VRAM.
        "--resolution=512",
        
        # CRITICAL: Batch size 1 is necessary to avoid OOM on 4GB VRAM.
        "--train_batch_size=1",
        
        # To compensate for the small batch size, I accumulate gradients over 4 steps.
        # This simulates a batch size of 4, making the training more stable.
        "--gradient_accumulation_steps=4",
        
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--max_train_steps=500", # 500 steps is usually the sweet spot for style transfer without overfitting.
        "--caption_column=text", 
        "--mixed_precision=fp16", # FP16 is faster and uses less memory than FP32.
        "--seed=42"
    ]
    
    print(f"[*] Executing command: {' '.join(command)}")
    
    # Firing off the training process.
    try:
        subprocess.run(command, check=True)
        print("[+] Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[!] Training failed with error: {e}")

if __name__ == "__main__":
    main()