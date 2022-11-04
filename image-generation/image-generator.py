from enum import Enum
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
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from time import sleep
from torch import autocast
from diffusers import StableDiffusionPipeline

MODELS = Enum("Model", ["dali", "stable_diffusion"])
DDB_TABLE_NAME = "stories"
IMAGE_OUTPUT_PATH = "images"
BUCKET_NAME = "story-images"
SQS_POLL_SLEEP_TIME = 60
IMAGE_SIZE = 256  # nothing else works without retraining the model maybe use this: https://github.com/alexjc/neural-enhance
DEFAULT_MODEL = MODELS.stable_diffusion
NUM_IMAGES_TO_GENERATE = 10

import boto3


def generate_path(path, chapter, image_num):
    image_path = path / f"{chapter}-{image_num}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    return image_path


def setup_stable_diffusion():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", use_auth_token=True
    ).to("cuda")

    def run_stable_diffusion(text, path, chapter):
        print(f"Running stable diffusion for chapter: {chapter} text: {text}")
        image_paths = []
        for i in range(NUM_IMAGES_TO_GENERATE):
            with autocast(
                "cuda",
            ):
                result = pipe(text)
                image = result.images[0]
                image_path = generate_path(path, chapter, i)
                image_paths.append(image_path)
                print("saving image to:" + str(image_path))
                image.save(image_path)
        return image_paths

    return run_stable_diffusion


def setup_dali_mini():
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
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)

    processor = DalleBartProcessor.from_pretrained(
        DALLE_MODEL, revision=DALLE_COMMIT_ID
    )

    dali_params = dict(
        p_generate=p_generate,
        p_decode=p_decode,
        key=key,
        processor=processor,
        params=params,
        vqgan_params=vqgan_params,
    )

    def run_dali(text, path, chapter):
        return generate_images_dali({chapter: text}, path, **dali_params)

    return run_dali


def generate_images_dali(
    texts_to_process: dict,
    path: Path,
    p_generate,
    p_decode,
    key,
    processor,
    params,
    vqgan_params,
    n_predictions=NUM_IMAGES_TO_GENERATE,
    gen_top_k=None,
    gen_top_p=None,
    temperature=None,
    cond_scale=10,
):
    # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
    text_to_file_dict = {v: k for k, v in texts_to_process.items()}
    image_paths = []
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
            decoded_images = decoded_images.clip(0.0, 1.0).reshape(
                (-1, IMAGE_SIZE, IMAGE_SIZE, 3)
            )
            for decoded_img in decoded_images:
                img = Image.fromarray(
                    np.asarray(decoded_img * (IMAGE_SIZE - 1), dtype=np.uint8)
                )
                save_to = generate_path(path, text_to_file_dict.get(text), i)
                img.save(save_to)
                print("saved image to:" + str(save_to))
                image_paths.append(save_to)
    return image_paths


def get_messages(queue):
    print("Starting to poll for messages...")
    while True:
        for message in queue.receive_messages(
            MessageAttributeNames=["story", "chapter"], MaxNumberOfMessages=10
        ):
            print("Recieved message")
            if not message or not message.body:
                print(f"Queue is empty, sleeping for {SQS_POLL_SLEEP_TIME} seconds")
                sleep(SQS_POLL_SLEEP_TIME)
                print("Done sleeping")
            else:
                yield dict(
                    text=message.body,
                    story_id=message.message_attributes.get("story").get("StringValue"),
                    chapter_id=message.message_attributes.get("chapter").get(
                        "StringValue"
                    ),
                    message=message,
                )


def upload_images(image_paths, story, s3):
    s3_paths = []
    for path in image_paths:
        # 1. upload to s3
        s3_path = f"stories/{story}/images/{path.name}"
        s3.upload_file(str(path), BUCKET_NAME, s3_path)
        s3_paths.append(s3_path)
    return s3_paths
    # 2. update dynamodb table for chapter, set new default image and mark the entry as not reviewed


def set_public_tag(s3_paths, s3):
    for path in s3_paths:
        s3.put_object_tagging(
            Bucket=BUCKET_NAME,
            Key=path,
            Tagging={"TagSet": [{"Key": "public", "Value": "yes"}]},
        )


def set_default_image(
    s3_path,
    story,
    chapter,
    ddb,
):
    ddb.update_item(
        TableName=DDB_TABLE_NAME,
        Key={
            "story": str(story),
            "chapter": str(chapter),
        },
        UpdateExpression="set image = :image",
        ExpressionAttributeValues={
            ":image": f"https://{BUCKET_NAME}.s3.eu-central-1.amazonaws.com/{s3_path}"
        },
    )


MODEL_SETUP_FUNCTIONS = {
    MODELS.dali: setup_dali_mini,
    MODELS.stable_diffusion: setup_stable_diffusion,
}


def init_model(new_model_enum, last_used_model_enum, previous_model):
    if last_used_model_enum == new_model_enum:
        return previous_model
    setup_func = MODEL_SETUP_FUNCTIONS.get(new_model_enum)
    return setup_func()


def main():
    model_to_use = DEFAULT_MODEL  # TODO select using the passed messages

    # 1. authenticate
    table = boto3.resource("dynamodb").Table(DDB_TABLE_NAME)
    sqs = boto3.resource("sqs")
    s3 = boto3.client("s3")
    queue = sqs.get_queue_by_name(QueueName="stories-texts-to-process.fifo")

    # 2. poll messages
    base_path = Path("/home/leonardo/Desktop/stories/dyno-staurie/stories")
    last_used_model = None
    model = None
    for i in get_messages(queue=queue):
        print("Deleting message from queue")
        try:
            i["message"].delete()
        except Exception as e:
            print("Unable to delete message")

        chapter = i["chapter_id"]
        story = i["story_id"]
        text = i["text"]
        print(f"Recieved message, story: {story}, chapter: {chapter}, text: {text}")
        story_path = base_path / f"{story}/images"
        # 3. process messages

        model = init_model(model_to_use, last_used_model, model)

        image_paths = model(text, story_path, chapter)
        last_used_model = model_to_use

        # 4. upload each image to s3
        s3_paths = upload_images(image_paths, story, s3)

        # change this to use best image when a smart way to detect this has been identified
        best_image = s3_paths[0]

        # 5. make the images publicly accessible
        set_public_tag(s3_paths, s3)

        # 6. update DDB
        print(f"Setting default image for story: {story} and chapter: {chapter}")
        set_default_image(best_image, story, chapter, table)
        print(f"Finished processing: {story} and chapter: {chapter}")


if __name__ == "__main__":
    main()
