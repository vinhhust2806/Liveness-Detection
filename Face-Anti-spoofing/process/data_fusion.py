import cv2
import sys
sys.path.append('/media/aivn24/partition1/Vinh/Zalo/CVPR19-Face-Anti-spoofing/')
from utils import *
sys.path.append('process/')
from data_helper import *
#from data_helper import *
from augmentation import *
#from utils import *
import pandas as pd

class FDDataset(Dataset):
    def __init__(self, mode, fold_index = None, image_size = 128, augment = None, balance = True,
      ):
        super(FDDataset, self).__init__()
        print('fold: '+str(fold_index))

        self.augment = augment
        self.mode = mode
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
        print(mode)
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
            if self.balance:
                self.train_list = transform_balance(self.train_list)
        
        elif self.mode == 'test':
            self.test_list = glob.glob('/content/images_public/images/*.jpg')
            self.num_data = len(self.test_list)

            

        print(self.num_data)

    def __getitem__(self, index):
        if self.mode == 'train':

            index = index % len(self.train_list)

            if self.balance:
                if random.randint(0,1)==0:
                    tmp_list = self.train_list[0]
                else:
                    tmp_list = self.train_list[1]

                pos = random.randint(0,len(tmp_list)-1)
                img_path = tmp_list[pos]
                label = 0 if 'fake' in img_path else 1
            else:
                img_path = self.train_list[index]
                label = 0 if 'fake' in img_path else 1

        elif self.mode == 'val':
            img_path = self.val_list[index]
            label = 0 if 'fake' in img_path else 1

        elif self.mode == 'test':
            img_path= self.test_list[index]
           

        color = cv2.imread(img_path,1)
        depth = cv2.imread(img_path.replace('images','depth'),1)
        #ir = cv2.imread(os.path.join(DATA_ROOT, ir),1)

        color = cv2.resize(color,(RESIZE_SIZE,RESIZE_SIZE))
        depth = cv2.resize(depth,(RESIZE_SIZE,RESIZE_SIZE))
        #ir = cv2.resize(ir,(RESIZE_SIZE,RESIZE_SIZE))

        if self.mode == 'train':
            color = color_augumentor(color,target_shape=(self.image_size, self.image_size, 3))
            depth = depth_augumentor(depth,target_shape=(self.image_size, self.image_size, 3))
            #ir = ir_augumentor(ir,target_shape=(self.image_size, self.image_size, 3))

            color = cv2.resize(color, (self.image_size, self.image_size))
            depth = cv2.resize(depth, (self.image_size, self.image_size))
            #ir = cv2.resize(ir, (self.image_size, self.image_size))

            image = np.concatenate([color.reshape([self.image_size, self.image_size, 3]),
                                    depth.reshape([self.image_size, self.image_size, 3])], axis=2)

            if random.randint(0, 1) == 0:
                random_pos = random.randint(0, 1)
                if random.randint(0, 1) == 0:
                    image[:, :, 3 * random_pos:3 * (random_pos + 1)] = 0
                else:
                    for i in range(2):
                        if i != random_pos:
                            image[:, :, 3 * i:3 * (i + 1)] = 0

            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([self.channels * 2, self.image_size, self.image_size])
            image = image / 255.0

            label = int(label)



            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

        elif self.mode == 'val':
            color = color_augumentor(color, target_shape=(self.image_size, self.image_size, 3),is_infer=True)
            depth = depth_augumentor(depth, target_shape=(self.image_size, self.image_size, 3),is_infer=True)
            #ir = ir_augumentor(ir, target_shape=(self.image_size, self.image_size, 3),is_infer=True)
            n = len(color)

            color = np.concatenate(color, axis=0)
            depth = np.concatenate(depth, axis=0)
            #ir = np.concatenate(ir, axis=0)

            image = np.concatenate([color.reshape([n,self.image_size, self.image_size, 3]),
                                    depth.reshape([n,self.image_size, self.image_size, 3])], axis=3)

            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels * 2, self.image_size, self.image_size])
            image = image / 255.0

            label = int(label)
            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

        elif self.mode == 'test':
            color = color_augumentor(color, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            depth = depth_augumentor(depth, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            #ir = ir_augumentor(ir, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            n = len(color)

            color = np.concatenate(color, axis=0)
            depth = np.concatenate(depth, axis=0)
            #ir = np.concatenate(ir, axis=0)

            image = np.concatenate([color.reshape([n, self.image_size, self.image_size, 3]),
                                    depth.reshape([n, self.image_size, self.image_size, 3]),
                                    ],
                                   axis=3)

            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels * 2, self.image_size, self.image_size])
            image = image / 255.0

            return torch.FloatTensor(image), test_id

    def __len__(self):
        return self.num_data

# check #################################################################
def run_check_train_data():
    dataset = FDDataset(mode = 'train')
    print(dataset)

    num = len(dataset)
    for m in range(num):
        i = np.random.choice(num)
        image, label = dataset[m]
        print(image.shape)
        print(label)

        if m > 100:
            break

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_train_data()


