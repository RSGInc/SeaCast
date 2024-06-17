import os
import h5py
import pandas as pd
import openmatrix as omx

import pandas as pd
import numpy as np
import os
import sys
import yaml
from collections import OrderedDict
import argparse
import subprocess

if __name__ == '__main__':

    # runtime args
    parser = argparse.ArgumentParser(prog='enplanement')
    parser.add_argument(
         '-c', '--configs',
         help = 'Config Directory')
    parser.add_argument(
        '-e', '--enplanements',
        help = 'Number of enplanements')
    parser.add_argument(
        '-n', '--nconnections',
        help = 'Number of connecting flights')
    
    args = parser.parse_args()
    config_dir = args.configs
    enplanements = int(args.enplanements)
    nconnections = int(args.nconnections)


    print('RUNNING ENPLANEMENT UPDATE!')
    with open(os.path.join(config_dir,'preprocessing.yaml.template')) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    # update number of enplanements
    update_enplanements = {'num_enplanements': enplanements, 'connecting': nconnections}
    settings.get('tours').update(update_enplanements)

    # write out the preprocessing yaml file.
    with open(os.path.join(config_dir, 'preprocessing.yaml'), 'w') as f:
        yaml.dump(settings, f) 
