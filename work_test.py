from argparse import ArgumentParser
from work_SSL import SSupervised
import os
from util import *

def parse_info() : 

    parser = ArgumentParser(description='Self-Supervised Test model')

    parser.add_argument('--img-path', help='\t\tTarget image path', required=True)
    parser.add_argument('--img-file-name', help='\t\tImage file name', required=True, type=str)
    
    parser.add_argument('--load-ckpt', help='load model checkpoint', required=True, type=str)
    parser.add_argument('--show-output', help='pop up window to display outputs', default=1, type=int)
    
    # ...   default options
    parser.add_argument('--resize', help='\t\tresizing image', default=800, type=int)
    parser.add_argument('--cnn-layer', help='\t\tDnCNN total layer', default=7, type=int)
    parser.add_argument('--noise-mode', help='\t\tNoise type', default='gaussian', type=str)
    parser.add_argument('--mask-mode', help='\t\tMask type', default='interpolate', type=str)
    parser.add_argument('--selector', default='test')
    return parser.parse_args()

if __name__ == '__main__' : 
    params = parse_info()
    print(f'\n\n** image in : {os.path.join(params.img_path, params.img_file_name)}\n\n')

    ssupervised = SSupervised(params)
    ssupervised.work_load_model(params.load_ckpt)
    test_loader = load_dataset(os.path.join(params.img_path, params.img_file_name), 0, params, shuffled=False, single=True)
    ssupervised.__start__(save_file = test_loader)

    