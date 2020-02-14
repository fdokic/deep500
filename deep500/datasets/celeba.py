"""
dataset link: https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
"""

import requests
import os.path

from deep500.lv2.dataset import FileListDataset
import numpy as np
from zipfile import ZipFile as zf
from PIL import Image

from deep500.utils.onnx_interop.losses import BCELoss

def celeba_shape():
    return (64, 64, 3)

def celeba_loss():
    return BCELoss

# IMPORTANT: download_celeba does not work anymore. No automated solution found yet
# Therefore: manually download & unzip contents into folder_path/img_align_celeba in order for the dataset to work
# Download URL: https://drive.google.com/uc?export=download&confirm=CGa5&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
def download_celeba_and_get_file_paths(folder=''):
    # adapted from: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    # 202599 images in total, 1.4GB in size
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    # old id: '0B7EVK8r0v71pZjFTYXZWM3FlRnM',
    response = session.get(URL, params={'id': '0B7EVK8r0v71pZjFTYXZWM3FlRnM'}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': '0B7EVK8r0v71pZjFTYXZWM3FlRnM', 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    destination = folder + '/celeba.zip'
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

    with zf(destination, 'r') as zipObj:
        data_files = zipObj.namelist()
        zipObj.extractall(folder)

        # test_files unnecessary, come from generator
        # training_files = [file for file in data_files if (int(file[0:-4]) % 5 != 0)]
        # test_files = [file for file in data_files if (int(file[0:-4]) % 5 == 0)]
    return data_files, []

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

class celeba_loader():
    def __init__(self, normalize=True, folder_path=''):
        self.normalize = normalize
        self.folder_path = folder_path

    def __call__(self, file_names):
        shape = celeba_shape()
        size = [len(file_names)]
        size += list(shape)
        batch = np.empty(size)

        for i, name in enumerate(file_names):
            img = Image.open(self.folder_path + '/' + name)
            img = img.resize((shape[0], shape[1]), Image.ANTIALIAS)
            if self.normalize:
                pix = np.array(img).astype(np.float32)
                for i in range(3):
                    pix[:, :, i] -= 0.5 # np.mean(pix[:, :, i])
                    pix[:, :, i] /= 0.5 # np.std(pix[:, :, i])
            else:
                pix = np.array(img).astype(np.float32)
            batch[i, :, :, :] = pix

        # reshape to (3, 64, 64) as needed by DCGAN and cast to floats
        batch = np.reshape(batch, (-1, 3, 64, 64)).astype(np.float32)
        return batch

class noise_loader():
    def __init__(self, sample_shape):
        self.sample_shape = sample_shape

    def __call__(self, indices):
        return np.random.normal(0, 1, shape=(len(indices), *self.sample_shape).astype(np.float32))

def load_noise(input_node, data_size, sample_shape):
    loader = noise_loader(sample_shape)
    return FileListDataset(list(range(0, data_size)), input_node, loader)

def load_celeba(input_node='', folder_path='', normalize=True):
    if not os.path.exists(folder_path + '/img_align_celeba'):
        SyntaxError('path {} does not exist. Download files manually in a folder_path/img_align_celeba'
                    .format(folder_path + '/img_align_celeba'))
    else:
        destination = (folder_path + '/img_align_celeba')
        data_files = np.asarray(os.listdir(destination))

        loader = celeba_loader(normalize, folder_path + '/img_align_celeba')

        return FileListDataset(data_files, input_node, loader)






