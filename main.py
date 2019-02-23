import numpy as np
import pandas as pd
from Ft_array import *
from describe import *
# from histogram import *
import argparse
from pandas.api.types import is_string_dtype


def main():
    parser = argparse.ArgumentParser(description='process linear regression')
    parser.add_argument('-s', '--scrypt', type=str, dest="scrypt", choices=['describe', 'histogram'],
    help="function scrypt")
    parser.add_argument(type=str, dest="dataset",
                             help="describe dataset")
        
    opt = parser.parse_args()
    if (opt.scrypt == 'describe'):
        describe(opt.dataset)

if __name__ == '__main__':
    main()
