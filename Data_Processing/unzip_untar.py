import zipfile
import tarfile
import os

def unzip_file(zip_path, put_zips_here):
    os.makedirs(put_zips_here, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(put_zips_here)

def untar_file(tar_path, put_tars_here):
    os.makedirs(put_tars_here, exist_ok=True)
    with tarfile.open(tar_path, 'r') as tar_ref:
        tar_ref.extractall(put_tars_here)