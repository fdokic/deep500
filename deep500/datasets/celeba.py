"""
dataset link: https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
"""

import requests
import os.path

from deep500.lv2.dataset import NumpyDataset
import numpy as np
from zipfile import ZipFile as zf
from PIL import Image


def download_celeba_and_get_file_paths(folder=''):
    # adapted from: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    # 202599 images in total, 1.4GB in size
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
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


"""
def d_open(path='/home/fdokic/Dokumente/entry_task/celeba/celeba.zip'):
    with zf(path,'r') as zipObj:
        data_files = zipObj.namelist()
        zipObj.extractall('/home/fdokic/Dokumente/entry_task/celeba/')
    return data_files
"""

if __name__ == "__main__":
    file_id = 'TAKE ID FROM SHAREABLE LINK'
    destination = 'DESTINATION FILE ON YOUR DISK'


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def load_celeba(input_node_name='', label_node_name='', *args, normalize=True,
                folder='', **kwargs):
    if (not os.path.exists(folder + '/img_align_celeba')):
        data_files, = download_celeba_and_get_file_paths(folder)
    else:
        destination = (folder + '/img_align_celeba')
        data_files = os.listdir(destination)

    train_set, label_set = _numpy_celeba(destination, data_files, normalization=normalize)
    train_set = np.array(train_set)
    label_set = np.array(label_set)
    return NumpyDataset(data=train_set, data_node=input_node_name), label_set


def _numpy_celeba(folder, data_files, normalization):
    width = 64
    height = 64
    test_images = []
    for file in data_files:
        img = Image.open(folder + '/' + file)
        img = img.resize((width, height), Image.ANTIALIAS)
        pix = np.array(img)
        if normalization:
            pix = np.array(img).astype(np.float32)
            for i in range(3):
                pix[:, :, i] -= np.mean(pix[:, :, i])
                pix[:, :, i] /= np.std(pix[:, :, i])
        test_images.append(pix)
    label_set = np.ones((len(test_images), 1))
    return test_images, label_set


def celeba_shape():
    return (64, 64, 3)


