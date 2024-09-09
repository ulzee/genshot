
import subprocess
import os
from flask import Flask, request, render_template_string, send_file, jsonify
from flask_cors import CORS
import requests
import argparse
from io import BytesIO
import json
from cryptography.fernet import Fernet
import base64

parser = argparse.ArgumentParser(description='Run the Flask server')
parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
parser.add_argument('--gen_port', type=int, default=7000, help='Port of the generation service')
args = parser.parse_args()

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/*": {"origins": "*"}})

media_dir = '/home/ubuntu/genshot/media'

with open('key', 'r') as key_file:
    key_string = key_file.read().strip()
secret_key = eval(key_string)

encoded_key = base64.urlsafe_b64encode(secret_key)
cipher_suite = Fernet(encoded_key)

@app.route('/up', methods=['GET'])
def up():
    return "up"

@app.route('/gen', methods=['POST', 'OPTIONS'])
def generate():
    prompt = request.form.get('prompt')
    batch_size = request.form.get('batch_size', 1, type=int)

    json_message = json.dumps({
        'prompt': prompt,
        'batch_size': batch_size
    })

    encrypted_message = cipher_suite.encrypt(json_message.encode())

    # Send GET request with encrypted message as a parameter
    response = requests.get(
        f'http://localhost:{args.gen_port}/gen',
        params={'body': encrypted_message}
    )
    response.raise_for_status()
    job_data = response.json()
    return json.dumps({'job_id': job_data['job_id']})

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    # Check for completed image
    completed_image_path = os.path.join(media_dir, f"image_{job_id}_completed.jpg")
    if os.path.exists(completed_image_path):
        # If completed image exists, return status completed
        return json.dumps({"status": "completed"})

    # Check for intermediate image in media folder
    progress_image_path = os.path.join(media_dir, f"image_{job_id}_progress.jpg")
    if os.path.exists(progress_image_path):
        # If progress image exists, return it
        # FIXME: handle path of media folder more reliably
        return send_file(progress_image_path, mimetype='image/jpeg')

    # If no images found, return a status indicating the job is still in progress
    print(progress_image_path)
    return json.dumps({"status": "in_progress"})

@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    completed_image_path = os.path.join(media_dir, f"image_{job_id}_completed.jpg")
    if os.path.exists(completed_image_path):
        return send_file(completed_image_path, mimetype='image/jpeg')
    return jsonify({"status": "not_found", "message": "Image not found"}), 404


if __name__ == '__main__':
    print('ready')
    app.run(host='0.0.0.0', port=args.port, debug=True)
