# -*- coding: utf-8 -*-
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
import requests
import os
from tqdm import tqdm
import shutil


def downloader(url, target_dir, resume_byte_pos=None):
    """Download url to disk with possible resumption.
    Parameters
    ----------
    resume_byte_pos: int
        Position of byte from where to resume the download
    """
    # Get size of file
    r = requests.head(url)
    file_size = int(r.headers.get('content-length', 0))

    # Append information to resume download at specific byte position
    # to header
    resume_header = ({'Range': f'bytes={resume_byte_pos}-'}
                     if resume_byte_pos else None)

    # Establish connection
    r = requests.get(url, stream=True, headers=resume_header)

    # Set configuration
    block_size = 1024
    initial_pos = resume_byte_pos if resume_byte_pos else 0
    mode = 'ab' if resume_byte_pos else 'wb'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    file = os.path.join(target_dir, url.split('/')[-1])

    with open(file, mode) as f:
        print('Downloading data from {}'.format(url))
        with tqdm(total=file_size, unit='B',
                  unit_scale=True, unit_divisor=1024,
                  desc='', initial=initial_pos,
                  ascii=True, miniters=1, leave=True) as pbar:
            for chunk in r.iter_content(32 * block_size):
                f.write(chunk)
                pbar.update(len(chunk))


def download_file(url, target_dir, extract=True):
    """Execute the correct download operation.
    Depending on the size of the file online and offline, resume the
    download if the file offline is smaller than online.
    """
    # Establish connection to header of file
    r = requests.head(url)

    # Get filesize of online and offline file
    file_size_online = int(r.headers.get('content-length', 0))
    file = os.path.join(target_dir, url.split('/')[-1])

    if os.path.exists(file):
        file_size_offline = os.stat(file).st_size
        if file_size_online != file_size_offline:
            downloader(url, target_dir, file_size_offline)
    else:
        downloader(url, target_dir)

    if extract:
        shutil.unpack_archive(file, target_dir)
