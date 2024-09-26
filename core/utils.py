import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
import os
import torch
import sys

from FastSAM.fastsam.prompt import FastSAMPrompt

# from ultralytics.models.fastsam.prompt import FastSAMPrompt
from os import listdir
from PIL import Image

int2color_map={
        0: (69, 117, 181),
        1: (145, 191, 219),
        2: (224, 243, 248),
        3: (254, 224, 144),
        4: (252, 141, 89),
        5: (215, 48, 39),
    }

color2int_map = {
    (69, 117, 181): 0,
    (145, 191, 219): 1,
    (224, 243, 248): 2,
    (254, 224, 144): 3,
    (252, 141, 89): 4,
    (215, 48, 39): 5,
}

category_ids={
    '(69, 117, 181)': 0,
    '(145, 191, 219)': 1,
    '(224, 243, 248)': 2,
    '(254, 224, 144)': 3,
    '(252, 141, 89)': 4,
    '(215, 48, 39)': 5,
}

def int2pixel(array):

    height, width = len(array), len(array[0])
    image = Image.new('RGB', (width, height))

    for y in range(height):
        for x in range(width):
            pixel_value = array[y][x]
            color = int2color_map[pixel_value]
            image.putpixel((x, y), color)

    return image

def pixel2int(image):

    reverse_color_map = {color: index for color, index in color2int_map.items()}

    # Convert image pixels to array of integer indices
    height, width, _ = image.shape
    integer_indices = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            pixel = tuple(image[y, x])
            index = reverse_color_map.get(pixel)
            integer_indices[y, x] = index if index is not None else 6  # Set to 6 if pixel not found in color map

    return integer_indices

def show_image(source, figsize=(12, 10), 
               cmap=None, axis='off', fig_args={},
                 show_args={}, **kwargs):
    
    source = cv2.imread(source)#[:, :, ::-1]
    plt.figure(figsize=figsize, **fig_args)
    plt.imshow(source, cmap=cmap, **show_args)
    plt.axis(axis)
    plt.show(**kwargs)

def normalize_seismic(array):

    normalized = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
    normalized = normalized.astype(np.uint8)
    seismic_norm = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    return seismic_norm


def normalize_annotation(segmentation, width, height):
    normalized_segmentation = [
        [x / width, y / height] for x, y in zip(segmentation[0][::2], segmentation[0][1::2])
    ]
    return [coord for pair in normalized_segmentation for coord in pair]


def getpreds_gts(p, model, img_size=640, split='val'):

    from os import path

    masks_path = sorted([path.join(p+'/masks/'+split, idx) for idx in listdir(p+'/masks/'+split)])
    masks = [Image.open(idx).convert('RGB') for idx in masks_path]

    images_path = sorted([path.join(p+'/images/'+split, idx) for idx in listdir(p+'/images/'+split)])
    images = [Image.open(idx).convert('RGB') for idx in images_path]

    gt_masks, pred_masks = [], []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for mask, image, path in zip(masks, images, masks_path):

        #process_prediction results
        results = model(image,
                    device=device,
                    retina_masks=True,
                    imgsz=img_size,
                    iou=0.7,
                    conf=0.3
                    )

        prompt = FastSAMPrompt(path, results).everything_prompt()

        class_id = []
        for r in results:
            for c in r.boxes.cls:
                class_id.append(int(c))
        x = {}

        for i, p in enumerate(prompt):
            x[class_id[i]] = p.cpu().numpy().astype(np.int8)

        pred_mask = dict(sorted(x.items()))
        pred_mask_ = [pred_mask[key] for key in sorted(pred_mask.keys())]
        pred_masks.append(pred_mask_)

        #process_groundtruths
        grayscale_indices = pixel2int(np.array(mask))
        unique_indices = np.unique(grayscale_indices)

        gt_mask = []

        for i, index in enumerate(unique_indices):
            if index <= 5:#number of classes
                mask = (grayscale_indices == index)
                gt_mask.append(mask)
        gt_masks.append(gt_mask)

    return pred_masks, gt_masks

