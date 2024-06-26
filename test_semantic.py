from copy import deepcopy
import torch
from argparse import ArgumentParser
from FastSAM.fastsam import FastSAM
from core.metrics import *
from core.utils import getpreds_gts

def FaciesSAM(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FastSAM(args.model_path)

    pred_masks, gt_masks = getpreds_gts(args.data_path, model, 
                                        img_size=args.img_sz, split=args.split)

    # sub_pred1 = deepcopy(pred_masks)
    # sub_gt1 = deepcopy(gt_masks) 
    #print semantic segmentation results
    mIoU(pred_masks, gt_masks, split=args.split)
    class_accuracy(pred_masks, gt_masks, split=args.split)
    pixel_accuracy(pred_masks, gt_masks, split=args.split)
    frequency_weighted_IU(pred_masks, gt_masks, split=args.split)


if __name__ == '__main__':
     
    parser = ArgumentParser(description='Hyperparameters')

    parser.add_argument('--model_path', nargs='?', type=str, default='FaciesSAM-x.pt',
                        help='model path')
    parser.add_argument('--data_path', nargs='?', type=str, default='data',
                        help='data path')
    parser.add_argument('--split', nargs='?', type=str, default='val',
                        help='val or test split')
    parser.add_argument('--img_sz', nargs='?', type=int, default=640,
                    help='Image size')

    args = parser.parse_args()

    FaciesSAM(args)
