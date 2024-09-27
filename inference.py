from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import time

# Función para cargar el pipeline del modelo
def load_pipeline(model_name):
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        revision="fp16",
        use_auth_token=True
    ).to("cuda")

# Función para generar una imagen a partir del texto
def text2image(prompt, repo_id):
    load_pipeline(repo_id)
    start = time.time()
    image = pipe(prompt).images[0]
    end = time.time()
    return image, start, end
