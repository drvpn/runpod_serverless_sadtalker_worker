'''
The MIT License (MIT)
Copyright © 2024 Dominic Powers

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import os
import requests
import shutil
import boto3
from botocore.config import Config

''' Convert string to boolean '''
def str2bool(v):
    return v.lower() in ("true", "1", "yes")

'''Downloads a file from a URL to a local path'''
def download_file(url, local_filename):
    try:
        print(f'[SadTalker]: Downloading {url}')
        if os.path.exists(local_filename):
            return local_filename, None
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return local_filename, None

    except Exception as e:
        return None, e

'''Uploads a file to an S3 bucket and makes it publicly readable.'''
def upload_to_s3(local_file, bucket_name, object_name):
    try:
        print(f'[SadTalker]: Uploading {object_name}')
        s3_client = boto3.client('s3',
                                 endpoint_url=os.getenv('BUCKET_ENDPOINT_URL'),
                                 aws_access_key_id=os.getenv('BUCKET_ACCESS_KEY_ID'),
                                 aws_secret_access_key=os.getenv('BUCKET_SECRET_ACCESS_KEY'),
                                 config=Config(signature_version='s3v4'))
        s3_client.upload_file(local_file, bucket_name, object_name, ExtraArgs={'ACL': 'public-read'})

        return f"{os.getenv('BUCKET_ENDPOINT_URL')}/{bucket_name}/{object_name}", None
    except Exception as e:
        return None, e

def sync_checkpoints():

    try:
        # Ensure the models are downloaded and available
        model_paths = [
            ('/app/SadTalker/checkpoints/mapping_00109-model.pth.tar', 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar'),
            ('/app/SadTalker/checkpoints/mapping_00229-model.pth.tar', 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar'),
            ('/app/SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors', 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors'),
            ('/app/SadTalker/checkpoints/SadTalker_V0.0.2_512.safetensors', 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors'),
            ('/app/SadTalker/gfpgan/weights/alignment_WFLW_4HG.pth', 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth'),
            ('/app/SadTalker/gfpgan/weights/detection_Resnet50_Final.pth', 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth'),
            ('/app/SadTalker/gfpgan/weights/GFPGANv1.4.pth', 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'),
            ('/app/SadTalker/gfpgan/weights/parsing_parsenet.pth', 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth')
        ]

        for local_path, url in model_paths:
            if not os.path.exists(local_path):
                result, error = download_file(url, local_path)

                if error:
                    return None, error

        return None, None

    except Exception as e:
        return None, e

''' Link to network volume to SadTalker and GFPGAN checkpoints (if present)'''
def map_network_volume():

    try:
        # Detect network volume mount point
        if os.path.exists('/runpod-volume'):

            network_volume_path = '/runpod-volume'

        elif os.path.exists('/workspace'):

            network_volume_path = '/workspace'

        else:
            # No network volume
            network_volume_path = None

        # Identify network volume
        if network_volume_path is None:
            print(f'[SadTalker]: No network volume detected, using ephemeral storage')
        else:
            print(f'[SadTalker]: Network volume detected at {network_volume_path}')

        if network_volume_path is not None:
            # Ensure the cache directory exists on network volume
            os.makedirs(f'{network_volume_path}/sadtalker/gfpgan/weights', exist_ok=True)
            os.makedirs(f'{network_volume_path}/sadtalker/checkpoints', exist_ok=True)

            # Ensure the gfpgan directory exists on the ephemeral storage
            os.makedirs(f'/app/SadTalker/gfpgan', exist_ok=True)

            # Remove existing .cache directory if it exists and create a symbolic link
            if os.path.islink('/app/SadTalker/gfpgan/weights') or os.path.exists('/app/SadTalker/gfpgan/weights'):
                if os.path.isdir('/app/SadTalker/gfpgan/weights'):
                    shutil.rmtree('/app/SadTalker/gfpgan/weights')

                else:
                    os.remove("/app/SadTalker/gfpgan/weights")

            if os.path.islink('/app/SadTalker/checkpoints') or os.path.exists('/app/SadTalker/checkpoints'):
                if os.path.isdir('/app/SadTalker/checkpoints'):
                    shutil.rmtree('/app/SadTalker/checkpoints')

                else:
                    os.remove("/app/SadTalker/checkpoints")

            # Create symlink to connect enhancer cache to network volume
            os.symlink(f'{network_volume_path}/sadtalker/gfpgan/weights', '/app/SadTalker/gfpgan/weights')
            os.symlink(f'{network_volume_path}/sadtalker/checkpoints', '/app/SadTalker/checkpoints')

        return None, None

    except Exception as e:
        return None, e
