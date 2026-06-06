from diffusers import StableDiffusionInpaintPipeline
import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting"
)

pipe = pipe.to("cuda")

def render_tryon(image, mask, prompt="fashion try-on realistic"):
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask
    ).images[0]
    return result
