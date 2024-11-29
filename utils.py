import os
import yaml
import numpy as np
import torch
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_config():
    config_name = f'MetaSVDD.yaml'
    config_path = f'./configs/{config_name}'
    with open(config_path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def build_dirs(config):
    num_imgs = config['num_imgs']
    backbone_ = config['backbone']
    file_name = f'MetaSVDD_{backbone_}_{num_imgs}'
    output_dir = f'./outputs/{file_name}'
    ckpt_dir = f'./checkpoints/{file_name}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)


    output_img_dir = f'{output_dir}/img'
    output_log_dir = f'{output_dir}/log'
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_log_dir):
        os.makedirs(output_log_dir)
    config['output_img_dir'] = output_img_dir
    config['output_log_dir'] = output_log_dir
    config['ckpt_dir'] = ckpt_dir
    config['file_name'] = file_name
    config['ckpt_dir_model'] = f'{ckpt_dir}/{file_name}.ckpt'