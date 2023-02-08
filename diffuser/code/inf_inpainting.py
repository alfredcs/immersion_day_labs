import base64
import torch
from io import BytesIO
from diffusers import EulerDiscreteScheduler, StableDiffusionInpaintPipeline


def model_fn(model_dir):
    # Load stable diffusion and move it to the GPU
    scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir, 
                                                   scheduler=scheduler,
                                                   revision="fp16",
                                                   torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()

    return pipe


def predict_fn(data, pipe):

    # get prompt & parameters
    prompt = data.pop("inputs", data)
    input_img = data.pop("input_img", data)
    mask_img = data.pop("mask_img", data)
    # set valid HP for stable diffusion
    num_inference_steps = data.pop("num_inference_steps", 25)
    guidance_scale = data.pop("guidance_scale", 6.5)
    num_images_per_prompt = data.pop("num_images_per_prompt", 2)
    image_length = data.pop("image_length", 512)

    # run generation with parameters
    generated_images = pipe(
        prompt,
        image = input_img,
        mask_image = mask_img,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        height=image_length,
        width=image_length,
    #)["images"] # for Stabel Diffusion v1.x
    ).images

    # create response
    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    # create response
    return {"generated_images": encoded_images}
