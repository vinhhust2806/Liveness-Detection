import cv2

import sys
sys.path.append('/media/aivn24/partition1/Vinh/Zalo/CVPR19-Face-Anti-spoofing/')
from utils import *
sys.path.append('process/')
from data_helper import *
import torch
import pandas as pd
from torch.utils.data.dataset import Dataset
RESIZE_SIZE = 512


class FDDataset(Dataset):
    def __init__(self, mode, modality='color', fold_index=-1, image_size=128, augment = None, augmentor = None, balance = True):
        super(FDDataset, self).__init__()
        print('fold: '+str(fold_index))
        print(modality)

        self.mode       = mode
        self.augment = augment
        self.balance = balance

        self.channels = 3
        self.train_image_file = r'/media/aivn24/partition1/Vinh/Zalo/train_frame.csv'
        self.val_image_file = r'/media/aivn24/partition1/Vinh/Zalo/val_frame.csv'
        self.image_size = image_size
        self.fold_index = fold_index

        self.set_mode(self.mode,self.fold_index)

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index
        print('fold index set: ', fold_index)

        if self.mode == 'val':
            self.val_list = list(pd.read_csv(self.val_image_file,header=None)[1])
            self.num_data = len(self.val_list)
            print('set dataset mode: val')

        elif self.mode == 'train':
            self.train_list = list(pd.read_csv(self.train_image_file,header=None)[1])

            random.shuffle(self.train_list)
            self.num_data = len(self.train_list)

            print('set dataset mode: train')
        elif self.mode == 'test':
            self.test_list = glob.glob('/content/images_public/images/*.jpg')
            self.num_data = len(self.test_list)
        #print(self.num_data)

    def __getitem__(self, index):

        if self.fold_index is None:
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return

        if self.mode == 'train':
          img_path = self.train_list[index]
          label = 0 if 'fake' in img_path else 1
        if self.mode == 'val':
          img_path = self.val_list[index]
          label = 0 if 'fake' in img_path else 1
        if self.mode == 'test':
          img_path = self.test_list[index]
        image = cv2.imread(img_path,1)
        image = cv2.resize(image,(RESIZE_SIZE,RESIZE_SIZE))

        if self.mode == 'train':
            image = self.augment(image, target_shape=(self.image_size, self.image_size, 3))

            image = cv2.resize(image, (self.image_size, self.image_size))
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([self.channels, self.image_size, self.image_size])
            image = image / 255.0
            label = int(label)

            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

        elif self.mode == 'val':
            image = self.augment(image, target_shape=(self.image_size, self.image_size, 3), is_infer = True)
            n = len(image)
            image = np.concatenate(image,axis=0)
            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels, self.image_size, self.image_size])
            image = image / 255.0
            label = int(label)

            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))
        
        elif self.mode == 'test':
            image = self.augment(image, target_shape=(self.image_size, self.image_size, 3), is_infer = True)
            n = len(image)
            image = np.concatenate(image,axis=0)
            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels, self.image_size, self.image_size])
            image = image / 255.0

            return torch.FloatTensor(image), torch.LongTensor(np.asarray(0).reshape([-1]))

    def __len__(self):
        return self.num_data


# check #################################################################
def run_check_train_data():
    from augmentation import color_augumentor
    augment = color_augumentor
    dataset = FDDataset(mode = 'train', fold_index=-1, image_size=32,  augment=augment)
    print(dataset)

    num = len(dataset)
    for m in range(num):
        i = np.random.choice(num)
        image, label = dataset[m]
        print(image.shape)
        print(label.shape)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_train_data()


