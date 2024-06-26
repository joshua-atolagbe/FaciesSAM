from pandas import DataFrame
from os import listdir
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import os
import shutil


def split_train_val(p, per_val, split='train'):
    
    df = DataFrame()
    df['images'] = sorted([idx for idx in listdir(p+'images/'+split)])
    df['masks'] = sorted([idx for idx in listdir(p+'masks/'+split)])
    df['labels'] = sorted([idx for idx in listdir(p+'labels/'+split)])

    #splitting into train and validation data
    trainset, validset = train_test_split(df, test_size=per_val, random_state=2024)
    
    return trainset, validset


def move_files(args):
    
    source_dir='/content/drive/My Drive/Facies Segmentation/data/data1/images/train'

    path = '/content/drive/My Drive/Facies Segmentation/data/data1/'
    val_images_path = path+'images/val'
    val_masks_path = path+'masks/val'
    val_labels_path = path+'labels/val'
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_masks_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)
    
    trainset, validset = split_train_val(path, args.per_val)

    #move images
    print('Splitting data into train and validation set')
    for filename in os.listdir(source_dir+'/images'):
        if filename in validset['images']:
            shutil.move(os.path.join(source_dir+'/images', filename), val_images_path)

    #move masks
    for filename in os.listdir(source_dir+'/masks'):
        if filename in validset['masks']:
            shutil.move(os.path.join(source_dir+'/masks', filename), val_masks_path)

    #move labels
    for filename in os.listdir(source_dir+'/labels'):
        if filename in validset['labels']:
            shutil.move(os.path.join(source_dir+'/labels', filename), val_labels_path)
    
    print('Done')

if __name__ == '__main__':

    parser = ArgumentParser(description='Hyperparameter')
    parser.add_argument('--per_val', nargs='?', type=float, default=None,
                        help='percentage of the validation')
    
    args = parser.parse_args()
    
    if args.per_val != None:
        move_files(args)
    else:
        raise NotImplementedError()
