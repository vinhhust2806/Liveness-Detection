from data import FDDataset
from augmentation import get_augment
augment = get_augment('color')
train_dataset = FDDataset(mode = 'train', modality='color',image_size=128,
                              fold_index=-1,augment=augment)
print(train_dataset[0])