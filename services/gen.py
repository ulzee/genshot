
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='flux')
parser.add_argument('--bridge_addr', type=str, default='54.70.9.212')
parser.add_argument('--port', type=int, default=5000)
parser.add_argument('--remote_port', type=int, default=5001)
parser.add_argument('--gpu', type=str, default=None)
args = parser.parse_args()

if args.gpu is not None:
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

# redis_conn = Redis()
# rdq = RedisQueue(connection='ec2-18-236-79-206.us-west-2.compute.amazonaws.com')

class CustomCallback(PipelineCallback):
    def __init__(self, job_id):
        self.job_id = job_id

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs):
        latents = callback_kwargs['latents']
        intermediate_images = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor).sample
        intermediate_images = (intermediate_images / 2 + 0.5).clamp(0, 1)
        intermediate_images = intermediate_images.cpu().permute(0, 2, 3, 1).float().numpy()
        intermediate_images = (intermediate_images * 255).round().astype("uint8")
        for idx, img in enumerate(intermediate_images):
            Image.fromarray(img).save(f"dump/{self.job_id}_intermediate_step_{step_index}_image_{idx}.jpg")
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
        try:
            images = pipe(
                [prompt] * batch_size,
                height=batch_size_config[batch_size],
                width=batch_size_config[batch_size],
                guidance_scale=3.5,
                num_inference_steps=32,
                max_sequence_length=512,
                callback_on_step_end=CustomCallback(job_id),
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
            redis_client = redis.Redis(host='ec2-18-236-79-206.us-west-2.compute.amazonaws.com', port=6379, db=0)
            redis_client.rpush('image_queue', img_byte_arr)
            # rdq.enqueue()
            # result = rdq.enqueue(count_words_at_url, 'http://nvie.com')

            image_status[job_id] = "completed"
        except Exception as e:
            image_status[job_id] = f"error: {str(e)}"
        finally:
            image_queue.task_done()

# Start the background thread
Thread(target=process_image_queue, daemon=True).start()

@app.route('/up', methods=['GET'])
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

# @app.route('/status/<job_id>', methods=['GET'])
# def check_status(job_id):
#     status = image_status.get(job_id, "not found")
#     return json.dumps({"job_id": job_id, "status": status}), 200

@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    if image_status.get(job_id) == "completed":
        image_path = f"media/{job_id}.jpg"
        if os.path.exists(image_path):
        #     with open(image_path, "rb") as image_file:
        #         encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            return send_file(image_path, mimetype='image/jpeg'), 200
    return jsonify({"status": "error", "message": image_status[job_id]}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if image_status.get(job_id) == "completed":
        # image_path = f"media/{job_id}.jpg"
        # if os.path.exists(image_path):
        #     with open(image_path, "rb") as image_file:
        #         encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        # else:
        #     return jsonify({"status": "error", "message": "Image file not found"}), 500
        return jsonify({
            "status": "completed",
        }), 200
    elif image_status.get(job_id) == "not found":
        return jsonify({"status": "not found", "message": "Job not found"}), 404
    elif image_status.get(job_id).startswith("error"):
        print(image_status[job_id])
        return jsonify({"status": "error", "message": image_status[job_id]}), 500
    else:
        # Find the latest intermediate image for this job_id
        intermediate_files = [f for f in os.listdir("dump") if f.startswith(f"{job_id}_intermediate_step_") and f.endswith(".jpg")]
        if intermediate_files:
            latest_file = max(intermediate_files, key=lambda x: int(x.split("_")[3]))
            return send_file(f"dump/{latest_file}", mimetype='image/jpeg'), 202
        else:
            return jsonify({"status": "starting" }), 202

if __name__ == '__main__':

    # Start SSH tunnel in the background using the bridge.pem key file
    subprocess.Popen(['ssh', '-N', '-R', f'{args.remote_port}:localhost:{args.port}', '-i', 'bridge.pem', f'ubuntu@{args.bridge_addr}'])

    mdl = "black-forest-labs/FLUX.1-dev"
    # mdl = "black-forest-labs/FLUX.1-schnell"
    if args.model == 'flux':
        pipe = FluxPipeline.from_pretrained(mdl, torch_dtype=torch.bfloat16, cache_dir='/data2/ulzee/hug')
    elif args.model == 'sd3m':
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir='/data2/ulzee/hug')
    if args.gpu is None:
        pipe.enable_model_cpu_offload()
    else:
        # FIXME:
        pipe.to(torch.device('cuda:0'))

    print('ready')
    app.run(host='0.0.0.0', debug=True, port=args.port)

