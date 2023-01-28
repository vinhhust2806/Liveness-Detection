import os,sys
import random
package_directory = os.path.dirname(os.path.abspath(__file__))
# print(package_directory[])
sys.path.append('/media/aivn24/partition1/Vinh/Zalo/CVPR19-Face-Anti-spoofing/')
from utils import *

DATA_ROOT = r'{path-to-dataset}/CASIA-SURF/phase1'
TRN_IMGS_DIR = DATA_ROOT + '/Training/'
TST_IMGS_DIR = DATA_ROOT + '/Val/'
RESIZE_SIZE = 512

def load_train_list():
    list = []
    f = open(DATA_ROOT + '/train_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_val_list():
    list = []
    f = open(DATA_ROOT + '/val_private_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_test_list():
    list = []
    f = open(DATA_ROOT + '/test_public_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)

    return list

def transform_balance(train_list):
    pos_list = []
    neg_list = []
    for tmp in train_list:
        if 'real' in tmp:
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    print(len(pos_list))
    print(len(neg_list))
    return [pos_list,neg_list]

def submission(probs, outname, mode='valid'):
    if mode == 'valid':
        f = open(DATA_ROOT + '/val_public_list.txt')
    else:
        f = open(DATA_ROOT + '/test_public_list.txt')

    lines = f.readlines()
    f.close()
    lines = [tmp.strip() for tmp in lines]

    f = open(outname,'w')
    for line,prob in zip(lines, probs):
        out = line + ' ' + str(prob)
        f.write(out+'\n')
    f.close()
    return list



