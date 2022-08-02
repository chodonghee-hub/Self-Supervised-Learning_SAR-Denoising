from argparse import ArgumentParser
from work_SSL import SSupervised
import os

def parse_info() : 

    parser = ArgumentParser(description='Self-Supervised Training model')

    parser.add_argument('--img-path', help='\t\tTarget image path', required=True)
    parser.add_argument('--img-file-name', help='\t\tImage file neme', required=True, type=str)
    parser.add_argument('--epoch', help='\t\tepoch', required=True, type=int)
    parser.add_argument('--lr', help='\t\tlearning rate', required=True, type=float)
    parser.add_argument('--resize', help='\t\tresizing image', default=800, type=int)
    parser.add_argument('--cnn-layer', help='\t\tDnCNN total layer', default=7, type=int)
    parser.add_argument('--noise-mode', help='\t\tNoise type', default='gaussian', type=str)
    parser.add_argument('--mask-mode', help='\t\tMask type', default='interpolate', type=str)
    return parser.parse_args()

if __name__ == '__main__' : 
    params = parse_info()
    print(f'\n\n** image in : {os.path.join(params.img_path, params.img_file_name)}\n\n')

    # epoch check 
    assert params.epoch%100 == 0, f'{params.epoch} % 1000 == {params.epoch%1000}'
    ssupervised = SSupervised(params)
    ssupervised.__start__()

    