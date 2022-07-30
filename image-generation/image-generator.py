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

DDB_TABLE_NAME = "stories"
IMAGE_OUTPUT_PATH = "images"
BUCKET_NAME = "story-images"
SQS_POLL_SLEEP_TIME = 60
IMAGE_SIZE = 256  # nothing else works without retraining the model maybe use this: https://github.com/alexjc/neural-enhance
import boto3


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
    seed = random.randint(0, 2 ** 32 - 1)
    key = jax.random.PRNGKey(seed)

    processor = DalleBartProcessor.from_pretrained(
        DALLE_MODEL, revision=DALLE_COMMIT_ID
    )

    return dict(
        p_generate=p_generate,
        p_decode=p_decode,
        key=key,
        processor=processor,
        params=params,
        vqgan_params=vqgan_params,
    )


def generate_images(
    texts_to_process: dict,
    path: Path,
    p_generate,
    p_decode,
    key,
    processor,
    params,
    vqgan_params,
    n_predictions=10,
    gen_top_k=None,
    gen_top_p=None,
    temperature=None,
    cond_scale=10,
):
    # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
    text_to_file_dict = {v: k for k, v in texts_to_process.items()}
    image_paths = []
    images = []
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
                images.append(img)
                save_to = path / f"{text_to_file_dict.get(text)}-{i}.png"
                save_to.parent.mkdir(parents=True, exist_ok=True)
                img.save(save_to)
                print("saved image to:" + str(save_to))
                image_paths.append(save_to)
    return image_paths, images


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
            decoded_images = decoded_images.clip(0.0, 1.0).reshape(
                (-1, IMAGE_SIZE, IMAGE_SIZE, 3)
            )
            for decoded_img in decoded_images:
                img = Image.fromarray(
                    np.asarray(decoded_img * (IMAGE_SIZE - 1), dtype=np.uint8)
                )
                save_to = path / f"{text_to_file_dict.get(text)}-{i}.png"
                img.save(save_to)
                print("saved image to:" + str(save_to))


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


def setup_clip():
    CLIP_REPO = "openai/clip-vit-base-patch32"
    CLIP_COMMIT_ID = None

    # Load CLIP
    clip, clip_params = FlaxCLIPModel.from_pretrained(
        CLIP_REPO, revision=CLIP_COMMIT_ID, dtype=jnp.float16, _do_init=False
    )
    clip_processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
    clip_params = replicate(clip_params)

    # score images
    @partial(jax.pmap, axis_name="batch")
    def p_clip(inputs, params):
        logits = clip(params=params, **inputs).logits_per_image
        return logits

    return {
        "clip_processor": clip_processor,
        "p_clip": p_clip,
        "clip_params": clip_params,
    }


def run_clip(prompts, images, clip_processor, p_clip, clip_params):
    # get clip scores
    clip_inputs = clip_processor(
        text=prompts * jax.device_count(),
        images=images,
        return_tensors="np",
        padding="max_length",
        max_length=77,
        truncation=True,
    ).data
    logits = p_clip(shard(clip_inputs), clip_params)

    # organize scores per prompt
    p = len(prompts)
    logits = np.asarray([logits[:, i::p, i] for i in range(p)]).squeeze()
    return logits[0].argsort()[::-1][0]


if __name__ == "__main__":
    # 1. authenticate
    table = boto3.resource("dynamodb").Table(DDB_TABLE_NAME)
    sqs = boto3.resource("sqs")
    s3 = boto3.client("s3")
    queue = sqs.get_queue_by_name(QueueName="stories-texts-to-process.fifo")

    # 2. poll messages
    dali_params = setup_dali_mini()
    # clip_params = setup_clip()
    base_path = Path("/home/leonardo/Desktop/stories/dyno-staurie/stories")
    for i in get_messages(queue=queue):
        chapter = i["chapter_id"]
        story = i["story_id"]
        text = i["text"]
        print(f"Recieved message, story: {story}, chapter: {chapter}, text: {text}")
        # 3. process messages
        image_paths, images = generate_images(
            texts_to_process={chapter: i["text"]},
            path=base_path / f"{story}/images",
            **dali_params,
        )
        # best_idx = run_clip([text], images, **clip_params)
        # print(f"Best image per CLIP score: {image_paths[best_idx]}")

        # 4. upload each image to s3
        s3_paths = upload_images(image_paths, story, s3)
        # change this to use best_idx when clip is working again
        best_image = s3_paths[0]
        # print(f"Best image in S3 per CLIP score: {best_image}")

        # 5. make the images publicly accessible
        set_public_tag(s3_paths, s3)

        # 6. update DDB
        print(f"Setting default image for story: {story} and chapter: {chapter}")
        set_default_image(best_image, story, chapter, table)

        # 7. delete message
        print("Deleting message from queue, processing finished")
        try:
            i["message"].delete()
        except Exception as e:
            print("Unable to delete message")
