import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, AutoencoderKL
import yaml
import os

def enrich_prompt(short_prompt):
    """
    Handles the prompt engineering logic. 
    I inject specific style keywords to ensure consistency with the game's art direction.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Hardcoding the visual style here. 
    # This ensures that even simple inputs like "cat" align with the game's isometric/3D aesthetic.
    base_instruction = "single character, centered, white background, mobile game style, cute but evil, 3d render, blender style, isometric, detailed texture"
    
    if api_key:
        try:
            # If an API key is present, I use GPT-4o to expand the prompt creatively while keeping constraints.
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a game asset AI. Output only the prompt."},
                    {"role": "user", "content": f"Create a Stable Diffusion prompt for: {short_prompt}. Ensure it is a single character, centered on white background. Style: {base_instruction}"}
                ]
            )
            return response.choices[0].message.content
        except:
            # Fallback to simple concatenation if the API call fails for any reason.
            return f"{short_prompt}, {base_instruction}, masterpiece"
    else:
        # Saving tokens/cost if no key is provided.
        return f"{short_prompt}, {base_instruction}, masterpiece, best quality, vibrant colors"

class VillainGenerator:
    def __init__(self, config_path="config.yaml"):
        # Handling path issues when running from different directories (e.g., via Streamlit)
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        try:
            with open(config_path, "r") as f:
                self.conf = yaml.safe_load(f)
            self.base_model = self.conf["model"]["base_model"]
            self.lora_path = self.conf["model"]["output_dir"]
        except:
            # If config is missing or broken, I default to DreamShaper v8 as it's reliable for 2.5D art.
            self.base_model = "Lykon/dreamshaper-8"
            self.lora_path = ""
        self.pipe = None
        
    def load_model(self):
        print("[*] Initializing pipeline on RTX 3050 (CUDA)...")
        
        # Swapping the default VAE for 'mse-vae'. 
        # The standard SD 1.5 VAE produces blurry eyes/details in 'float16'. 
        # Using MSE fixes the sharpness without needing full float32 precision.
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", 
            torch_dtype=torch.float16 
        )
        
        # EulerDiscrete is faster and converges better for this style than the default PNDM.
        scheduler = EulerDiscreteScheduler.from_pretrained(self.base_model, subfolder="scheduler")
        
        # Loading the main pipeline.
        # Disabling the safety checker to prevent false positives (black images) on innocent game assets.
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model, 
            vae=vae, 
            scheduler=scheduler,
            torch_dtype=torch.float16, # Keeping everything in half-precision to save VRAM.
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if os.path.exists(self.lora_path):
            print(f"[*] Found LoRA weights at {self.lora_path}, loading...")
            try:
                self.pipe.load_lora_weights(self.lora_path)
            except Exception as e:
                print(f"[!] Failed to load LoRA: {e}")
        
        # --- MEMORY OPTIMIZATION STRATEGY ---
        try:
            import accelerate
            print("[*] Accelerate library detected. Enabling CPU offload.")
            # This is critical for my 4GB VRAM GPU. 
            # It offloads unused model parts to RAM, preventing OOM errors.
            self.pipe.enable_model_cpu_offload()
            
        except ImportError:
            print("[!] Accelerate not found. Falling back to standard CUDA load.")
            self.pipe.to("cuda")
            # If offload isn't available, slicing attention helps reduce peak memory usage.
            self.pipe.enable_attention_slicing()

        except Exception as e:
            print(f"[!] Offload failed ({e}). forcing fallback.")
            self.pipe.to("cuda")
            self.pipe.enable_attention_slicing()
            
    def generate(self, prompt, num_images=1, use_llm=True, steps=40): 
        # Lazy loading: I only load the heavy model into memory when the user actually requests an image.
        if not self.pipe:
            self.load_model()
            
        final_prompt = enrich_prompt(prompt) if use_llm else prompt
        print(f"[*] Processing: {final_prompt}")
        
        # I added specific negative prompts like 'grid' and 'sprite sheet' because 
        # SD 1.5 tends to generate multiple small characters otherwise.
        negative_prompt = "bad anatomy, blurry, low quality, cropped, text, watermark, messy, glitch, noise, pattern, texture, many characters, grid, sprite sheet"
        
        images = self.pipe(
            final_prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            num_images_per_prompt=num_images,
            guidance_scale=7.5 # 7.5 is the sweet spot for DreamShaper.
        ).images
        
        return images, final_prompt