from torch import autocast
from diffusers import StableDiffusionPipeline


def setup_stable_diffusion():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", use_auth_token=True
    ).to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    with autocast(
        "cuda",
    ):
        result = pipe(prompt)
        print(result)
        image = result.images[0]

    image.save("astronaut_rides_horse.png")
