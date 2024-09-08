from flask import Flask, request, render_template_string, send_file
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
cors = CORS(app)

with open('key', 'r') as key_file:
    key_string = key_file.read().strip()
secret_key = eval(key_string)

encoded_key = base64.urlsafe_b64encode(secret_key)
cipher_suite = Fernet(encoded_key)

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
        f'http://localhost:{args.gen_port}/up',
        params={'body': encrypted_message}
    )
    response.raise_for_status()
    job_data = response.json()
    return json.dumps({'job_id': job_data['job_id']})

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    response = requests.get(f'http://localhost:{args.gen_port}/status/{job_id}')
    response.raise_for_status()
    if response.headers.get('content-type') == 'application/json':
        return json.dumps(response.json())
    else:
        return send_file(BytesIO(response.content), mimetype='image/jpeg')

@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    response = requests.get(f'http://localhost:{args.gen_port}/result/{job_id}')
    response.raise_for_status()
    return send_file(BytesIO(response.content), mimetype='image/jpeg')


if __name__ == '__main__':
    print('ready')
    app.run(host='0.0.0.0', port=args.port, debug=True)
