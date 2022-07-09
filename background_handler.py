from pathlib import Path
import jax
import jax.numpy as jnp
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
from functools import partial
import random
from dalle_mini import DalleBartProcessor
from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange
from flax.jax_utils import replicate
import sys

IMAGE_OUTPUT_PATH = "images"


def run_dali_mini(texts_to_process: dict, path: Path):
    # Model references

    # dalle-mega
    DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
    DALLE_COMMIT_ID = None

    # if the notebook crashes too often you can use dalle-mini instead by uncommenting below line
    # DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"

    # VQGAN model
    VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
    VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

    # Load dalle-mini
    model, params = DalleBart.from_pretrained(
        DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
    )

    # Load VQGAN
    vqgan, vqgan_params = VQModel.from_pretrained(
        VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
    )

    params = replicate(params)
    vqgan_params = replicate(vqgan_params)

    # model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(
        tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
    ):
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    # decode image
    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        return vqgan.decode_code(indices, params=params)

    # create a random key
    seed = random.randint(0, 2 ** 32 - 1)
    key = jax.random.PRNGKey(seed)

    processor = DalleBartProcessor.from_pretrained(
        DALLE_MODEL, revision=DALLE_COMMIT_ID
    )

    # number of predictions per prompt
    n_predictions = 10

    # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 10.0

    text_to_file_dict = {v: k for k, v in texts_to_process.items()}
    for text in texts_to_process.values():
        prompts = [text]
        tokenized_prompts = processor(prompts)
        tokenized_prompt = replicate(tokenized_prompts)

        print(f"Processing: {text}\n")
        for i in range(max(n_predictions // jax.device_count(), 1)):
            print("Processing batch:", i)
            # get a new key
            key, subkey = jax.random.split(key)
            # generate images
            encoded_images = p_generate(
                tokenized_prompt,
                shard_prng_key(subkey),
                params,
                gen_top_k,
                gen_top_p,
                temperature,
                cond_scale,
            )
            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]
            # decode images
            decoded_images = p_decode(encoded_images, vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for decoded_img in decoded_images:
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                save_to = path / f"{text_to_file_dict.get(text)}-{i}.png"
                img.save(save_to)
                print("saved image to:" + str(save_to))


if __name__ == "__main__":
    run_dali_mini(
        texts_to_process={
            "1": "Once upon a time there was a young boy",
            "2": "He lived in a small wooden house in a large forest",
            "3": "One night, the boy woke up from a loud cracking sound just outside the house.",
            "3.1": "It was dark outside so he could not see anything",
            "3.2": "A few hours later he woke up to a sunny morning",
            "FINAL_CHAPTER": "The boy saw that the old oak tree had fallen over and inside sat a small girl with purple eyes. Then the boy woke up.",
        },
        path=Path(
            "/home/leonardo/Desktop/stories/dyno-staurie/stories/test_story_1/images"
        ),
    )
