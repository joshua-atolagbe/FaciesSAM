import numpy as np
import os
import cv2
from core.utils import normalize_seismic, int2pixel
from tqdm.auto import tqdm
from PIL import Image

splits = ['train', 'val', 'test']

def load_data():
    
    train_path = '/content/drive/My Drive/Facies Segmentation/data/data0/train'
    test_path = '/content/drive/My Drive/Facies Segmentation/data/data0/test_once'

    train_s = np.load(train_path+'/train_seismic.npy')
    train_l = np.load(train_path+'/train_labels.npy')

    test1_s = np.load(test_path+'/test1_seismic.npy')
    test1_l = np.load(test_path+'/test1_labels.npy')

    test2_s = np.load(test_path+'/test2_seismic.npy')
    test2_l = np.load(test_path+'/test2_labels.npy')

    trains = [train_s, train_l]
    tests1 = [test1_s, test1_l]
    tests2 = [test2_s, test2_l]


    return trains, tests1, tests2

def save_images_masks(path, data, split:str='train'):

    masks = 'masks'; images='images'

    if split == 'train':

        train_images_path = path+'/'+images+'/'+split
        train_masks_path = path+'/'+masks+'/'+split
        os.makedirs(train_images_path, exist_ok=True)
        os.makedirs(train_masks_path, exist_ok=True)

        # save inline
        print('Saving train images and masks')
        for idx in tqdm(range(data[0].shape[0])):
            cv2.imwrite(f'{train_images_path}/image_inline_{idx+300}.jpg', normalize_seismic(data[0][idx, :, :].T))
            int2pixel(data[1][idx, :, :].T).save(f'{train_masks_path}/mask_inline_{idx+300}.png')
                            # data[1][idx, :, :].T.astype(np.uint16))

        # save xline
        for idx in tqdm(range(data[0].shape[1])):
            cv2.imwrite(f'{train_images_path}/image_xline_{idx+300}.jpg', normalize_seismic(data[0][:, idx,:].T))
            int2pixel(data[1][:, idx, :].T).save(f'{train_masks_path}/mask_xline_{idx+300}.png')
                            # data[1][:, idx, :].T.astype(np.uint16))

        return 'Done'

    elif split == 'val':

        test1_images_path = path+'/'+images+'/'+split
        test1_masks_path = path+'/'+masks+'/'+split
        os.makedirs(test1_images_path, exist_ok=True)
        os.makedirs(test1_masks_path, exist_ok=True)

        # save inline
        print('Saving #test1 images and masks')
        for idx in tqdm(range(data[0].shape[0])):
            cv2.imwrite(f'{test1_images_path}/test1_image_inline_{idx+100}.jpg', normalize_seismic(data[0][idx, :, :].T))
            int2pixel(data[1][idx, :, :].T).save(f'{test1_masks_path}/test1_mask_inline_{idx+100}.png')
                            # data[1][idx, :, :].T.astype(np.uint16))

        # save xline
        for idx in tqdm(range(data[0].shape[1])):
            cv2.imwrite(f'{test1_images_path}/test1_image_xline_{idx+300}.jpg', normalize_seismic(data[0][:, idx,:].T))
            int2pixel(data[1][:, idx, :].T).save(f'{test1_masks_path}/test1_mask_xline_{idx+300}.png')
                            # data[1][:, idx, :].T.astype(np.uint16))

        return 'Done'

    elif split == 'test':

        test2_images_path = path+'/'+images+'/'+split
        test2_masks_path = path+'/'+masks+'/'+split
        os.makedirs(test2_images_path, exist_ok=True)
        os.makedirs(test2_masks_path, exist_ok=True)

        # save inline
        print('Saving #test2 images and masks')
        for idx in tqdm(range(data[0].shape[0])):
            cv2.imwrite(f'{test2_images_path}/test2_image_inline_{idx+100}.jpg', normalize_seismic(data[0][idx, :, :].T))
            int2pixel(data[1][idx, :, :].T).save(f'{test2_masks_path}/test2_mask_inline_{idx+100}.png')
                            # data[1][idx, :, :].T.astype(np.uint16))

        # save xline
        for idx in tqdm(range(data[0].shape[1])):
            cv2.imwrite(f'{test2_images_path}/test2_image_xline_{idx+1001}.jpg', normalize_seismic(data[0][:, idx,:].T))
            int2pixel(data[1][:, idx, :].T).save(f'{test2_masks_path}/test2_mask_xline_{idx+1001}.png'),
                            # data[1][:, idx, :].T.astype(np.uint16))

        return 'Done' 

if __name__ == '__main__':
    trains, tests1, tests2 = load_data()
    path = '/content/drive/My Drive/Facies Segmentation/data/data1'
    save_images_masks(path, data=trains, split='train')
    save_images_masks(path, data=tests1, split='val')
    save_images_masks(path, data=tests2, split='test')
