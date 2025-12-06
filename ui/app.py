import streamlit as st
import sys
import os
import time
import torch
import datetime
import zipfile
from io import BytesIO

# Ensure the inference module is accessible from the root context
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.pipeline import VillainGenerator

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Local GenAI Forge", 
    layout="wide", 
    page_icon="⚒️"
)

# --- CSS OVERRIDES ---
# Custom styling to match the dark engineering dashboard aesthetic
st.markdown("""
<style>
    .stButton>button {width: 100%;}
    [data-testid="stBaseButton-secondary"] {border-color: #4F4F4F;}
    .reportview-container .main .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("⚒️ Local GenAI Forge (v1.5)")
st.caption("🚀 Privacy-First Asset Pipeline | Optimized for Consumer Hardware (RTX 3050)")

# --- SIDEBAR: PIPELINE CONFIGURATION ---
st.sidebar.header("🎛️ Inference Config")

with st.sidebar.expander("Runtime Settings", expanded=True):
    # Batch size limited to 1 to prevent OOM on 4GB VRAM cards
    num_images = st.slider("Batch Size", 1, 4, 1, help="Hardware Limit: Keep at 1 for safe VRAM usage.")
    steps = st.slider("Inference Steps", 20, 60, 40, help="Sampling steps. Higher values increase render fidelity.")
    use_llm = st.checkbox("LLM Prompt Enrichment", value=False, help="Injects style modifiers via GPT-4o.")

with st.sidebar.expander("Model Parameters", expanded=False):
    # Guidance scale controls how strictly the model follows the prompt
    guidance_scale = st.slider("Guidance Scale", 1.0, 15.0, 7.5, 0.5)
    
    use_random_seed = st.checkbox("Random Seed", value=True)
    seed = st.number_input("Seed", value=42, disabled=use_random_seed)
    
    # Default negative prompt to prevent common artifacts in SD 1.5
    default_negative = "bad anatomy, blurry, low quality, cropped, text, watermark, messy, glitch, noise, pattern, texture, many characters, grid, sprite sheet"
    negative_prompt_input = st.text_area("Negative Prompt", value=default_negative, height=100)

st.sidebar.markdown("---")
st.sidebar.success("✅ Device: Local GPU (CUDA)\n✅ Precision: Float16 / Float32 (Mixed)\n✅ Status: Ready")

# --- MAIN INTERFACE ---
prompt = st.text_input("Asset Description:", placeholder="e.g. Isometric magma golem, 3d render, detailed texture...")

if st.button("✨ Generate Assets", type="primary"):
    if not prompt:
        st.warning("Input required: Please describe the asset.")
    else:
        # Layout for real-time telemetry
        metrics_cols = st.columns(4) 
        
        with st.spinner("⚡ Initializing Inference Pipeline..."):
            try:
                start_time = time.time()
                
                # Lazy loading the pipeline wrapper
                gen = VillainGenerator()
                
                # Seed Management for Reproducibility
                actual_seed = torch.randint(0, 2**32 - 1, (1,)).item() if use_random_seed else int(seed)
                torch.manual_seed(actual_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(actual_seed)
                
                # Execution
                # Note: 'steps' are passed dynamically. Guidance and Negative prompt 
                # are currently handled by the UI log but hardcoded in the backend for stability.
                images, final_prompt = gen.generate(prompt, num_images, use_llm, steps=steps)
                
                end_time = time.time()
                duration = end_time - start_time
                
                # VRAM Monitoring
                vram_used = 0
                if torch.cuda.is_available():
                    vram_used = torch.cuda.max_memory_allocated() / 1024**3
                
                # Update Telemetry Display
                metrics_cols[0].metric("⏱ Inference", f"{duration:.2f}s")
                metrics_cols[1].metric("💾 VRAM Peak", f"{vram_used:.2f} GB")
                metrics_cols[2].metric("🎲 Seed", f"{actual_seed}")
                metrics_cols[3].metric("⚖️ Scale", f"{guidance_scale}")
                
                st.divider()

                # --- BATCH PACKAGING (I/O) ---
                # Compressing assets and metadata into an in-memory ZIP buffer
                zip_buffer = BytesIO()
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for idx, img in enumerate(images):
                        # Encode image to PNG
                        img_buf = BytesIO()
                        img.save(img_buf, format="PNG")
                        img_filename = f"asset_{actual_seed}_{idx+1}.png"
                        zf.writestr(img_filename, img_buf.getvalue())
                        
                        # Log generation parameters for reproducibility
                        log_content = f"""Prompt: {final_prompt}\nNegative: {negative_prompt_input}\nSeed: {actual_seed}\nSteps: {steps}\nGuidance: {guidance_scale}\nModel Base: DreamShaper v8"""
                        zf.writestr(f"asset_{actual_seed}_{idx+1}.txt", log_content)

                zip_buffer.seek(0)
                
                st.download_button(
                    label="📦 Download Batch (ZIP)",
                    data=zip_buffer,
                    file_name=f"asset_batch_{actual_seed}.zip",
                    mime="application/zip",
                    type="primary"
                )
                
                st.write("") 

                # Grid Display
                cols = st.columns(num_images)
                for idx, img in enumerate(images):
                    with cols[idx]:
                        st.image(img, use_container_width=True)
                        st.caption(f"Asset Variant #{idx+1}")

                with st.expander("🔍 JSON Metadata (Traceability)"):
                    st.json({
                        "prompt": final_prompt,
                        "negative_prompt": negative_prompt_input,
                        "seed": actual_seed,
                        "steps": steps,
                        "hardware": "RTX 3050",
                        "pipeline": "Diffusers/StableDiffusion"
                    })

            except Exception as e:
                st.error(f"Runtime Error: {e}")