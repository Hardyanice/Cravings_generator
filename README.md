---
title: Food Image Generator
emoji: 💻
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
short_description: Stable diffusion-based finetuned generator for food images
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---

🍽️ AI Food Image Generator
Generate stunning, realistic food images using AI! This application uses fine-tuned Stable Diffusion models specialized for Western and Indian cuisine to create quality food photos from simple text descriptions.

✨ Features:

🍔 Western Cuisine Model: Specialized in burgers, pizza, pasta, steaks, salads, and more

🍛 Indian Cuisine Model: Expert in curry, biryani, dosa, tandoori, and traditional dishes

⚡ Moderate speed: Generate images in 2-3 minutes (optimized with Euler scheduler)

🖥️ Easy Interface: Simple web-based UI, no technical knowledge required

🔧 Customizable: Adjustable steps, guidance scale, and seed for reproducible results


<b>How to Use (Online)</b>

A. Direct Website Access

1. Visit our live demo: https://huggingface.co/spaces/Hardyanice/Food_image_generator
   
2. Choose your cuisine: Click either "Western Cuisine" or "Indian Cuisine"

3. Describe your food: Enter a detailed description like:

"A juicy beef burger with lettuce, tomato, and fries on the side"
"Butter chicken curry with basmati rice and fresh naan bread"


4. Generate: Click the "Generate Image" button

5. Download: Right-click the generated image to save


B. Self-Hosting your own model(Duplicate Space)

1. Log in to your Hugging Face account.

2. Go to this Space: Food_Image_Generator.

3. Click "Duplicate this Space" (top-right corner).

4. Choose:

Space name (e.g., my-food-generator)

Visibility (public/private)

Hardware (GPU recommended, like T4 or A10G)

5. Click "Create Space" — Hugging Face will copy all files and automatically install dependencies.