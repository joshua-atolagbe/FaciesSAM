import torch
from argparse import ArgumentParser
from ultralytics import YOLO

def FaciesSAM(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(args.model_path) 
    results = model.val(data=args.cfg, split=args.split,
                            device=device, plots=True,
                            imgsz=args.img_sz)

    print('Done')

if __name__ == '__main__':
     
    parser = ArgumentParser(description='Hyperparameters')

    parser.add_argument('--model_path', nargs='?', type=str, default='FaciesSAM-x.pt',
                        help='model path')
    parser.add_argument('--cfg', nargs='?', type=str, default='sa.yaml',
                        help='Data configuration file')
    parser.add_argument('--split', nargs='?', type=str, default='val',
                        help='val or test split')
    parser.add_argument('--img_sz', nargs='?', type=int, default=640,
                    help='Image size')

    args = parser.parse_args()

    FaciesSAM(args)

