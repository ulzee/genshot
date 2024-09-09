import redis
import os
import time
from PIL import Image
import io

# Redis connection
redis_client = redis.Redis(host='35.165.222.103', port=6379, db=0)

# Ensure the media directory exists
media_dir = '/home/ubuntu/genshot/media'
os.makedirs(media_dir, exist_ok=True)

def save_image(image_data, filename):
    image = Image.open(io.BytesIO(image_data))
    image.save(os.path.join(media_dir, filename))
    print(f"Saved image: {filename}")

def main():
    print("Starting image saver...")
    while True:
        # Check for new entries with prefix 'image_queue_'
        keys = redis_client.keys('image_queue_*')
        for key in keys:
            image_data = redis_client.lpop(key)
            if image_data:
                jobid, stage = key.decode().split('_')[2:2+2]
                filename = f"image_{jobid}_{stage}.jpg"
                save_image(image_data, filename)
                # Remove the key if it's empty
                # if redis_client.llen(key) == 0:
                redis_client.delete(key)

        # Small delay to prevent excessive CPU usage
        time.sleep(0.01)

if __name__ == "__main__":
    main()
