import numpy as np  
import torch
from ultralytics import YOLO
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
    parser.add_argument('--cfg', nargs='?', type=str, default='data/sa.yaml',
                        help='Data configuration file')
    parser.add_argument('--model_path', nargs='?', type=str, default='models/FaciesSAM-x.pt',
                        help='Path to pretrained model')
    parser.add_argument('--name', nargs='?', type=str, default='faciesam-s',
                        help='model name')                        

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
        name=args.name+'_'+str(args.img_sz)+'_',
        imgsz=args.img_sz,
        save=True,
        optimizer='SGD',
        overlap_mask=False,
        val=True,
        augment=args.aug,
        boxes=False,
        patience=50,
        plots=True,
        fliplr= 0.5,
        mosaic= 1.0, #you can modify
        mixup= 0.15, #you can modify
        copy_paste= 0.3, #you can modify
        scale=0.9, #you can modify
    )

