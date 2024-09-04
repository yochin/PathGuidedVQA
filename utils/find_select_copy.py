import os
import sys
import argparse
import yaml
import logging

import random
import shutil
from glob import glob
from PIL import Image, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(description='Copy image files')   

    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()

    return args

def set_path_logger(path_to_log):
    if not os.path.exists(os.path.split(path_to_log)[0]):
        os.makedirs(os.path.split(path_to_log)[0])

    logFormatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s', datefmt='%Y/%m/%d %p %I:%M:%S, ')
    logFileHandler = logging.FileHandler(path_to_log)
    logConsoleHandler = logging.StreamHandler(sys.stdout)

    logFileHandler.setFormatter(logFormatter)
    logConsoleHandler.setFormatter(logFormatter)

    logging.getLogger().addHandler(logFileHandler)
    logging.getLogger().addHandler(logConsoleHandler)
    logging.getLogger().setLevel(logging.DEBUG)

    
def find_images_in_directory(directory):
    # 이미지 파일 확장자 목록
    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.JPG']

    # 하위 디렉토리를 포함하여 모든 이미지 파일을 검색
    image_files = []
    for extension in img_extensions:
        image_files.extend(glob(os.path.join(directory, '**', extension), recursive=True))
    return image_files

# def copy_random_images(image_files, destination_folder, num_images):
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)

#     # 이미지 파일 리스트 중에서 랜덤으로 선택
#     selected_images = random.sample(image_files, min(num_images, len(image_files)))
#     logging.info(f'num selected images: {len(selected_images)}')

#     for img_path in selected_images:
#         try:
#             img = Image.open(img_path)
#             img = ImageOps.exif_transpose(img)
#             img.verify()  # 이미지가 손상되지 않았는지 확인
#             shutil.copy(img_path, destination_folder)
#             logging.info(f'Success copying {img_path}')
#         except Exception as e:
#             logging.info(f"Error copying {img_path}: {e}")

def copy_random_images(image_files, destination_folder, num_images, ref_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    random.shuffle(image_files)

    # 참조 폴더에 있는 이미지를 확인
    ref_images = set(os.listdir(ref_folder))

    selected_images = []
    for img_path in image_files:
        if len(selected_images) >= num_images:
            break
        img_name = os.path.basename(img_path)
        if img_name in ref_images:
            logging.info(f'Skipping {img_path} as it exists in reference folder')
            continue
        selected_images.append(img_path)
    
    logging.info(f'num selected images: {len(selected_images)}')

    for img_path in selected_images:
        try:
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            img.verify()  # 이미지가 손상되지 않았는지 확인
            shutil.copy(img_path, destination_folder)
            logging.info(f'Success copying {img_path}')
        except Exception as e:
            logging.info(f"Error copying {img_path}: {e}")

def main():
    args = parse_args()

    with open(args.config) as fid:
        conf = yaml.load(fid, Loader=yaml.FullLoader)

    # set a log file path        
    path_to_log = os.path.join(conf['output_dir'], conf['task_name'], 'list_select_copy.log')
    set_path_logger(path_to_log)

    logging.info('list_select_copy.py: Start')
    logging.info(conf)

    if conf['image_server']['copy_to_db_images']:
        # do copy images
        src_dir = conf['image_server']['path']
        dst_dir = os.path.join(conf['db']['base_dir'], conf['db']['images'])
        ref_dir = conf['image_server']['ref_path']

        num_images = conf['image_server']['n_images']

        logging.info(f'Start to find images in directory: {src_dir}')
        image_files = find_images_in_directory(src_dir)
        logging.info(f'Found {len(image_files)} images in directory: {src_dir}')

        copy_random_images(image_files, dst_dir, num_images, ref_dir)

    logging.info('list_select_copy.py: Finish')


if __name__ == '__main__':
    main()

    
