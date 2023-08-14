# parse arguments

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Param')

    # path
    parser.add_argument('--path', type=str, default='', help='path to config')
    parser.add_argument('--prefix', type=str, default='', help='prefix for the experiment')
    parser.add_argument('--new_dir', default=False, type=int, help='If True, concat datetime string to exp_dir.')
    
    # device
    parser.add_argument('--cpu_workers', default=1, type=int,
                        help='number of cpu workers for each agent')    
    parser.add_argument('--gpu', default=-1, type=int,
                        help='will set CUDA_VISIBLE_DEVICES to selected value')
    parser.add_argument('--seed', default=-1, type=int,
                        help='overrides config/default seed for more convenient seed setting.')
    
    # logging
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # example
    parser.add_argument('--count', type=int, default=1, help='Number of times to repeat')
    # parser.add_argument('mode', choices=['read', 'write'],  default='read', help='Specify mode (read or write)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    return parser.parse_args()