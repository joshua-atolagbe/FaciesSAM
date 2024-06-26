from os import path
import numpy as np
import os
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon
from os import listdir
from utils import category_ids, normalize_annotation
from tqdm.auto import tqdm
from getData import splits

def create_sub_masks(mask_image):

    width, height = mask_image.size

    sub_masks = {}
    for x in range(width):
        for y in range(height):
            pixel = mask_image.getpixel((x,y))[:3]

            if pixel != (0, 0, 0):
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks, width, height


def create_sub_mask_annotation(sub_mask):

    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation='low')

    segmentations = []
    for contour in contours:
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        # polygons.append(poly)
        segmentation = np.array(poly.exterior.coords)
        segmentation = np.maximum(segmentation, 0).ravel().tolist()
        segmentations.append(segmentation)

    return segmentations


def get_data_info(p, split):
    
    assert split in splits, f'Unidentified split'
    masks_path = sorted([path.join(p+'masks/'+split, idx) for idx in listdir(p+'masks/'+split)])
    # images_path = sorted([path.join(p+split+'/images', idx) for idx in listdir(p+split+'/images')])
    mask_images = [Image.open(idx) for idx in masks_path]
    images_file = sorted([idx for idx in listdir(p+'images/'+split)])

    return  mask_images, images_file


def min_index(arr1, arr2):

    #https://github.com/chrise96/image-to-coco-json-converter/blob/master/src/create_annotations.py

    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

def merge_multi_segment(segments):
    
    #https://github.com/chrise96/image-to-coco-json-converter/blob/master/src/create_annotations.py

    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

def save_labels(p='/content/drive/My Drive/Facies Segmentation/data/data1/', split='train'):

    labels = p+'labels/'+split

    os.makedirs(labels, exist_ok=True)

    mask_images, images_file = get_data_info(p=p, split=split)

    # Create the annotations
    print(f'Saving #{split} labels')
    for mask_image, file in tqdm(zip(mask_images, images_file)):
        sub_masks, width, height = create_sub_masks(mask_image)

        annotations = []

        for color, sub_mask in sub_masks.items():
            category_id = category_ids[color]
            annotation = create_sub_mask_annotation(sub_mask)
            if len(annotation) != 1:
                #filter empty annotation
                annotation = [subannot for subannot in annotation if subannot]
                #combine multiple polygons of same class and instance
                if len(annotation) != 1:
                    annotation = merge_multi_segment(annotation)
                    annotation = [[item for subannot1 in annotation \
                                   for subannot2 in subannot1 for item in subannot2]]

            annotation = normalize_annotation(annotation, width, height)
            annotations.append([int(category_id)]+annotation)

        #save annotation
        with open(f'{labels}/{file[:-4]}.txt', mode='w') as f:
            f.write('\n'.join([' '.join(map(str, sublist)) for sublist in annotations]))

    return 'Done'

if __name__ == '__main__':
    save_labels(split='train')
    save_labels(split='val')
    save_labels(split='test')
