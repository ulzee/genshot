
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='flux')
parser.add_argument('--bridge_addr', type=str, default='54.70.9.212')
parser.add_argument('--port', type=int, default=5000)
parser.add_argument('--remote_port', type=int, default=5001)
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--num_steps', type=int, default=32)
parser.add_argument('--cache_dir', type=str, default='/u/scratch/u/ulzee/hug')
args = parser.parse_args()

if args.gpu is not None and 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


import subprocess
import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from diffusers.callbacks import PipelineCallback
from time import time
from flask import Flask, request, send_file, jsonify
from io import BytesIO
from cryptography.fernet import Fernet
from flask_cors import CORS
import json
from PIL import Image
import math
import base64
import uuid
from threading import Thread
from queue import Queue
import io
import numpy as np
# from rq import Queue as RedisQueue
# from redis import Redis
import redis

with open('key', 'r') as key_file:
    key_string = key_file.read().strip()
secret_key = eval(key_string)

encoded_key = base64.urlsafe_b64encode(secret_key)
f = Fernet(encoded_key)

app = Flask(__name__)
CORS(app)

batch_size_config = {
    1: 1024,
    4: 512,
    9: 256
}

class CustomCallback(PipelineCallback):
    def __init__(self, job_id, **kwargs):
        self.job_id = job_id
        self.kwargs = kwargs

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs):
        latents = callback_kwargs['latents']
        if args.model == 'sd3m':
            intermediate_images = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor).sample
            intermediate_images = (intermediate_images / 2 + 0.5).clamp(0, 1)
            intermediate_images = intermediate_images.cpu().permute(0, 2, 3, 1).float().numpy()
            intermediate_images = (intermediate_images * 255).round().astype("uint8")
        elif 'FLUX' in args.model:
            latents = pipeline._unpack_latents(latents, self.kwargs['height'], self.kwargs['width'], pipeline.vae_scale_factor)
            latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
            intermediate_images = [
                pipeline.image_processor.postprocess(image, output_type='pil')[0] \
                    for image in pipeline.vae.decode(latents, return_dict=False)]
            intermediate_images = np.array([np.asarray(i) for i in intermediate_images])

        # Combine images into a square grid
        grid_size = math.ceil(math.sqrt(len(intermediate_images)))
        single_image_size = intermediate_images[0].shape[:2]
        grid_image = Image.new('RGB', (single_image_size[1] * grid_size, single_image_size[0] * grid_size))

        for idx, img in enumerate(intermediate_images):
            x = (idx % grid_size) * single_image_size[1]
            y = (idx // grid_size) * single_image_size[0]
            grid_image.paste(Image.fromarray(img), (x, y))

        # Convert the combined image to bytes
        img_byte_arr = io.BytesIO()
        grid_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Send the combined image bytes to Redis queue
        redis_client = redis.Redis(host='35.165.222.103', port=6379, db=0)
        redis_client.rpush(f'image_queue_{self.job_id}_progress', img_byte_arr)
        redis_client.close()

        return dict()


# Create a queue for background processing
image_queue = Queue()

# Dictionary to store image generation status
image_status = {}

def process_image_queue():
    while True:
        job = image_queue.get()
        if job is None:
            break
        job_id, prompt, batch_size = job
        # try:
        images = pipe(
            [prompt] * batch_size,
            height=batch_size_config[batch_size],
            width=batch_size_config[batch_size],
            guidance_scale=3.5,
            num_inference_steps=args.num_steps,
            max_sequence_length=512,
            callback_on_step_end=CustomCallback(
                job_id,
                height=batch_size_config[batch_size],
                width=batch_size_config[batch_size],
            ),
        ).images

        # Join the images into one large square image
        grid_size = math.ceil(math.sqrt(len(images)))
        total_width = images[0].width * grid_size
        total_height = images[0].height * grid_size
        combined_image = Image.new('RGB', (total_width, total_height))

        for idx, image in enumerate(images):
            x = (idx % grid_size) * images[0].width
            y = (idx // grid_size) * images[0].height
            combined_image.paste(image, (x, y))


        # Convert the image to bytes
        img_byte_arr = io.BytesIO()
        combined_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Send the image bytes to Redis queue
        redis_client = redis.Redis(host='35.165.222.103', port=6379, db=0)
        redis_client.rpush(f'image_queue_{job_id}_completed', img_byte_arr)
        redis_client.close()

        image_status[job_id] = "completed"

# Start the background thread
Thread(target=process_image_queue, daemon=True).start()

@app.route('/up', methods=['GET'])
def up():
    return 'up'

@app.route('/gen', methods=['GET'])
def generate_image():
    data = request.args
    encrypted_body = data.get('body', '')
    if not encrypted_body:
        return "No encrypted body provided", 400
    try:
        decrypted_body = f.decrypt(encrypted_body.encode()).decode()
        body_json = json.loads(decrypted_body)
        prompt = body_json.get('prompt', 'cat')
        batch_size = int(body_json.get('batch_size', 1))
    except Exception as e:
        return f"Error decrypting or parsing body: {str(e)}", 400
    if not prompt:
        return "No prompt provided", 400

    if batch_size < 1:
        return "Invalid batch size", 400

    # Generate a unique ID for this job
    job_id = str(uuid.uuid4())

    # Add the job to the queue
    image_queue.put((job_id, prompt, batch_size))

    # Set initial status
    image_status[job_id] = "queued"

    return json.dumps({"job_id": job_id, "status": "queued"}), 200

if __name__ == '__main__':

    # Start SSH tunnel in the background using the bridge.pem key file
    subprocess.Popen(['ssh', '-N', '-R', f'{args.remote_port}:localhost:{args.port}', '-i', 'bridge.pem', f'ubuntu@{args.bridge_addr}'])

    if args.model == 'sd3m':
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir=args.cache_dir)
    elif 'FLUX' in args.model:
        pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir)

    pipe.to(torch.device('cuda:0'))
    # pipe.enable_model_cpu_offload()

    print('ready')
    app.run(host='0.0.0.0', debug=True, port=args.port)

