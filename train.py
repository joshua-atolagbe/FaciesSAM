import numpy as np  
import torch
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr
from argparse import ArgumentParser

random_seed = 1234
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
np.random.seed(random_seed)

#freeze layer [optional]
#-----------------------------------------------------------------------------------------------
def freeze_layer(trainer, num):
    model = trainer.model
    num_freeze = num
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  #layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")

#Augmentation
#-----------------------------------------------------------------------------------------------
def __init__(self, p=1.0):
    """Initialize the transform object for YOLO bbox formatted params."""
    self.p = p
    self.transform = None
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A

        # check_version(A.__version__, "1.0.3", hard=True)  # version requirement

        # Transforms
        T = [
                # A.Rotate(limit = 10, p=0.5),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.01),
                A.RandomGamma(p=0.01),
                A.ImageCompression(quality_lower=75, p=0.01),
                A.MotionBlur(p=0.01, blur_limit=(3,9), always_apply=False),
                # A.HorizontalFlip(p=0.01),
        ]
        self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
    except ImportError:  # package not installed, skip
        pass
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")

Albumentations.__init__ = __init__


if __name__ == '__main__':
     
    parser = ArgumentParser(description='Hyperparameters')

    parser.add_argument('--num_freeze', nargs='?', type=int, default=0,
                        help='Number of layers to freeze')
    parser.add_argument('--aug', nargs='?', type=bool, default=False,
                        help='Whether to use data augmentation.')
    parser.add_argument('--epochs', nargs='?', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--bs', nargs='?', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img_sz', nargs='?', type=int, default=640,
                        help='Image size')
    parser.add_argument('--cfg', nargs='?', type=str, default='sa.yaml',
                        help='Data configuration file')
    parser.add_argument('--model_path', nargs='?', type=str, default='models/FaciesSAM-x.pt',
                        help='Path to pretrained model')

    args = parser.parse_args()

    model = YOLO(args.model_path)

    if args.num_freeze > 0:
        model.add_callback('on_train_start', lambda trainer: freeze_layer(trainer, num=args.num_freeze))

    model.train(
        data=args.cfg,
        task='segment',
        mode='train',
        epochs=args.epochs,
        batch=args.bs,
        imgsz=args.img_sz,
        save=True,
        optimizer='SGD',
        overlap_mask=False,
        val=True,
        augment=args.aug,
        boxes=False,
        patience=15,
        plots=True,
        fliplr= 0.5,
        mosaic= 1.0,
        mixup= 0.15,
        copy_paste= 0.3,
        scale=0.9,
    )


