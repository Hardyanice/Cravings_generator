import gradio as gr
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import os

# Try to import accelerate
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("⚠️ Accelerate library not available. Some optimizations will be disabled.")

# ---------- CONFIG ----------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_WESTERN = "Western_model"  # Directory path, not file
LORA_INDIAN = "Indian_model"    # Directory path, not file
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Fixed inference parameters for optimal performance
INFERENCE_STEPS = 8
GUIDANCE_SCALE = 5

# Global variables for pipelines
base_pipeline = None
pipeline_western = None
pipeline_indian = None
active_pipeline = None

# ---------- MODEL LOADING FUNCTIONS ----------
def load_base_model():
    """Load the base Stable Diffusion model with optimizations"""
    global base_pipeline
    
    print("Loading base Stable Diffusion model...")
    try:
        base_pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL, 
            torch_dtype=torch_dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Use faster scheduler
        base_pipeline.scheduler = EulerDiscreteScheduler.from_config(
            base_pipeline.scheduler.config,
            use_karras_sigmas=True
        )
        
        base_pipeline.to(device)
        
        # Enable memory optimizations
        if hasattr(base_pipeline, "enable_attention_slicing"):
            base_pipeline.enable_attention_slicing(1)
            print("✅ Attention slicing enabled")
        
        if hasattr(base_pipeline, "enable_vae_slicing"):
            base_pipeline.enable_vae_slicing()
            print("✅ VAE slicing enabled")
        
        # Enable xFormers if available
        if hasattr(base_pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                base_pipeline.enable_xformers_memory_efficient_attention()
                print("✅ xFormers memory efficient attention enabled")
            except Exception as e:
                print(f"⚠️ xFormers not available: {e}")
        
        # Enable CPU offload only if accelerate is available and we have GPU
        if ACCELERATE_AVAILABLE and device == "cuda" and hasattr(base_pipeline, "enable_model_cpu_offload"):
            try:
                base_pipeline.enable_model_cpu_offload()
                print("✅ Model CPU offload enabled")
            except Exception as e:
                print(f"⚠️ CPU offload failed: {e}")
        elif device == "cpu":
            print("ℹ️ Running on CPU - skipping CPU offload optimization")
        
        # Enable sequential CPU offload as fallback if accelerate is available
        if ACCELERATE_AVAILABLE and hasattr(base_pipeline, "enable_sequential_cpu_offload"):
            try:
                base_pipeline.enable_sequential_cpu_offload()
                print("✅ Sequential CPU offload enabled")
            except Exception as e:
                print(f"⚠️ Sequential CPU offload failed: {e}")
            
        print(f"✅ Base model loaded on {device}")
        return base_pipeline
        
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        return None

def load_lora_pipeline(lora_path, cuisine_name):
    """Load LoRA weights into a pipeline copy"""
    if base_pipeline is None:
        print("❌ Base pipeline not loaded!")
        return None
        
    print(f"Loading {cuisine_name} LoRA from {lora_path}...")
    
    try:
        # Check if LoRA directory exists
        if not os.path.exists(lora_path):
            print(f"❌ LoRA path not found: {lora_path}")
            return None
            
        # Create a copy of the base pipeline
        pipe_copy = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL, 
            torch_dtype=torch_dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Load LoRA weights using the correct method
        pipe_copy.load_lora_weights(lora_path)
        
        # Use faster scheduler
        pipe_copy.scheduler = EulerDiscreteScheduler.from_config(
            pipe_copy.scheduler.config,
            use_karras_sigmas=True
        )
        
        pipe_copy.to(device)
        
        # Enable all available optimizations
        if hasattr(pipe_copy, "enable_attention_slicing"):
            pipe_copy.enable_attention_slicing(1)
        
        if hasattr(pipe_copy, "enable_vae_slicing"):
            pipe_copy.enable_vae_slicing()
        
        # Enable xFormers if available
        if hasattr(pipe_copy, "enable_xformers_memory_efficient_attention"):
            try:
                pipe_copy.enable_xformers_memory_efficient_attention()
            except:
                pass
        
        # Enable CPU offload only if accelerate is available and conditions are met
        if ACCELERATE_AVAILABLE and device == "cuda" and hasattr(pipe_copy, "enable_model_cpu_offload"):
            try:
                pipe_copy.enable_model_cpu_offload()
            except:
                pass
        
        # Enable sequential CPU offload as fallback
        if ACCELERATE_AVAILABLE and hasattr(pipe_copy, "enable_sequential_cpu_offload"):
            try:
                pipe_copy.enable_sequential_cpu_offload()
            except:
                pass
            
        print(f"✅ {cuisine_name} LoRA loaded successfully")
        return pipe_copy
        
    except Exception as e:
        print(f"❌ Error loading {cuisine_name} LoRA: {e}")
        print(f"Make sure {lora_path} contains the LoRA adapter files")
        return None

def initialize_models():
    """Initialize all models at startup"""
    global base_pipeline, pipeline_western, pipeline_indian
    
    # Load base model first
    base_pipeline = load_base_model()
    if base_pipeline is None:
        return False
    
    # Load LoRA pipelines
    pipeline_western = load_lora_pipeline(LORA_WESTERN, "Western")
    pipeline_indian = load_lora_pipeline(LORA_INDIAN, "Indian")
    
    # Check if at least one LoRA loaded successfully
    if pipeline_western is None and pipeline_indian is None:
        print("❌ No LoRA models loaded successfully!")
        return False
        
    print("✅ Model initialization complete!")
    print(f"Fixed settings - Steps: {INFERENCE_STEPS}, Guidance: {GUIDANCE_SCALE}")
    return True

# ---------- GENERATION FUNCTION ----------
def generate_image(prompt, seed, progress=gr.Progress()):
    """Generate image using the active pipeline with optimized settings"""
    global active_pipeline
    
    if active_pipeline is None:
        return None, "❌ No cuisine selected! Please go back and select a cuisine type."
    
    if not prompt or prompt.strip() == "":
        return None, "❌ Please enter a prompt!"
    
    try:
        progress(0.1, desc="Preparing generation...")
        
        # Set up generator
        if seed == -1:
            generator = torch.Generator(device=device)
        else:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        
        progress(0.3, desc="Generating image...")
        
        # Generate with fixed optimized settings and additional optimizations
        with torch.inference_mode():
            # Use torch.cuda.empty_cache() to free up memory if on CUDA
            if device == "cuda":
                torch.cuda.empty_cache()
            
            # Enable autocast for mixed precision (faster inference) - only if on CUDA
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch_dtype):
                    result = active_pipeline(
                        prompt=prompt,
                        num_inference_steps=INFERENCE_STEPS,  # Fixed to 8
                        guidance_scale=GUIDANCE_SCALE,        # Fixed to 15.0
                        generator=generator,
                        return_dict=True
                    )
            else:
                # CPU inference without autocast
                result = active_pipeline(
                    prompt=prompt,
                    num_inference_steps=INFERENCE_STEPS,  # Fixed to 8
                    guidance_scale=GUIDANCE_SCALE,        # Fixed to 15.0
                    generator=generator,
                    return_dict=True
                )
        
        progress(1.0, desc="Complete!")
        
        info = f"✅ Generated successfully!\nOptimized Settings - Steps: {INFERENCE_STEPS}, Guidance: {GUIDANCE_SCALE}"
        if seed != -1:
            info += f", Seed: {seed}"
            
        return result.images[0], info
        
    except Exception as e:
        return None, f"❌ Generation failed: {str(e)}"

# ---------- BUTTON HANDLERS ----------
def choose_cuisine(cuisine):
    """Select cuisine and set active pipeline"""
    global active_pipeline
    
    if cuisine == "western":
        if pipeline_western is not None:
            active_pipeline = pipeline_western
            cuisine_name = "🍔 Western Cuisine Selected"
        else:
            return gr.update(visible=True), gr.update(visible=False), "❌ Western model not available"
    else:
        if pipeline_indian is not None:
            active_pipeline = pipeline_indian
            cuisine_name = "🍛 Indian Cuisine Selected"
        else:
            return gr.update(visible=True), gr.update(visible=False), "❌ Indian model not available"
    
    return gr.update(visible=False), gr.update(visible=True), cuisine_name

def go_back():
    """Return to landing page"""
    global active_pipeline
    active_pipeline = None
    return gr.update(visible=True), gr.update(visible=False), ""

# ---------- BUILD UI ----------
def create_interface():
    """Create the Gradio interface with custom styling"""
    
    # Custom CSS for better visual appeal
    custom_css = """
    .main-title {
        text-align: center !important;
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-size: 3rem !important;
        margin-bottom: 10px !important;
    }
    .cuisine-image img {
        height: 300px !important;
        object-fit: cover !important;
        border-radius: 15px !important;
        width: 100% !important;
    }
    .cuisine-container {
        margin: 20px 0 !important;
    }
    .optimized-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 10px !important;
        border-radius: 8px !important;
        text-align: center !important;
        margin: 10px 0 !important;
    }
    """
    
    with gr.Blocks(title="AI Food Generator", theme=gr.themes.Soft(), css=custom_css) as demo:
        
        # Landing Page
        with gr.Column(visible=True) as landing_page:
            gr.Markdown("# 🍽️ AI Food Image Generator", elem_classes=["main-title"])
            gr.Markdown("### Choose your cuisine to get started!", elem_id="subtitle")
            
            # Use Gradio's native components with custom styling
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=300):
                    # Western Cuisine
                    western_img = gr.Image(
                        "https://images.unsplash.com/photo-1551782450-a2132b4ba21d?w=600&h=400&auto=format&fit=crop&q=80",
                        show_label=False,
                        interactive=False,
                        container=True,
                        elem_classes=["cuisine-image"]
                    )
                    gr.Markdown("##  Jayesh's Food 101")
                    gr.Markdown("*Burgers, Pizza, Pasta, Steaks*")
                    btn_west = gr.Button(
                        "🍔 Select Western Cuisine", 
                        variant="primary", 
                        size="lg",
                        visible=True
                    )
                    if pipeline_western is None:
                        gr.Markdown("❌ Western model not available")

                with gr.Column(scale=1, min_width=300):
                    # Indian Cuisine
                    indian_img = gr.Image(
                        "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=600&h=400&auto=format&fit=crop&q=80",
                        show_label=False,
                        interactive=False,
                        container=True,
                        elem_classes=["cuisine-image"]
                    )
                    gr.Markdown("##  Souhardya's Khane ki Duniya")
                    gr.Markdown("*Curry, Biryani, Dosa, Tandoori*")
                    btn_ind = gr.Button(
                        "🍛 Select Indian Cuisine", 
                        variant="primary", 
                        size="lg",
                        visible=True
                    )
                    if pipeline_indian is None:
                        gr.Markdown("❌ Indian model not available")

        # Generation Page
        with gr.Column(visible=False) as generation_page:
            cuisine_title = gr.Markdown("## 🍽️ Generate Food Image(It will take anywhere between 2 to 4 minutes to generate an image.)")
            
            with gr.Row():
                with gr.Column(scale=1):
                    prompt = gr.Textbox(
                        label="📝 Describe the food you want",
                        placeholder="e.g., A delicious burger with fries, professional food photography",
                        lines=3
                    )
                    
                    # Only seed control remains - steps and guidance are now fixed
                    seed = gr.Number(
                        value=-1, 
                        label="🎲 Seed (-1 for random)",
                        info="Use the same seed to reproduce results"
                    )
                    
                    btn_generate = gr.Button("🎨 Generate Image", variant="primary", size="lg")
                    btn_back = gr.Button("⬅️ Back to Home", variant="secondary")
                
                with gr.Column(scale=1):
                    output_img = gr.Image(label="Generated Image", type="pil")
                    generation_info = gr.Markdown("")
        
        # Event handlers
        def safe_choose_cuisine(cuisine):
            try:
                return choose_cuisine(cuisine)
            except Exception as e:
                return gr.update(visible=True), gr.update(visible=False), f"❌ Error: {str(e)}"
        
        def safe_generate(prompt, seed):
            try:
                return generate_image(prompt, seed)
            except Exception as e:
                return None, f"❌ Error: {str(e)}"
        
        # Connect events - Using the visible buttons directly
        btn_west.click(
            lambda: safe_choose_cuisine("western"),
            outputs=[landing_page, generation_page, cuisine_title]
        )
        
        btn_ind.click(
            lambda: safe_choose_cuisine("indian"),
            outputs=[landing_page, generation_page, cuisine_title]
        )
        
        btn_generate.click(
            safe_generate,
            inputs=[prompt, seed],  # Removed steps and guidance inputs
            outputs=[output_img, generation_info]
        )
        
        btn_back.click(
            go_back,
            outputs=[landing_page, generation_page, cuisine_title]
        )
    
    return demo

# ---------- MAIN ----------
if __name__ == "__main__":
    print("🚀 Starting AI Food Image Generator...")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Accelerate available: {ACCELERATE_AVAILABLE}")
    print(f"Optimized settings - Steps: {INFERENCE_STEPS}, Guidance: {GUIDANCE_SCALE}")
    
    # Print additional system info
    if device == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    else:
        print("⚠️ Running on CPU - consider using GPU for faster inference")
    
    # Initialize models
    if not initialize_models():
        print("❌ Failed to initialize models. Please check your model files.")
        print("Make sure the following directories exist:")
        print(f"  - {LORA_WESTERN}/")
        print(f"  - {LORA_INDIAN}/")
        print("And contain the LoRA adapter files.")
        exit(1)
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )