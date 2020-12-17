from argparse import ArgumentParser

from mmdet.apis import (inference_detector, init_detector, 
                        show_result_pyplot, json_generation,
                        json_generation_float)
from PIL import Image
import glob
import os

image_list = []

        
def main():
    parser = ArgumentParser()
    #parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
 

        
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    for img in glob.glob('./single_images/*.jpg'): #assuming gif
        im=Image.open(img)
        image_list.append(im)
        file_name=os.path.basename(img)
        #print(file_name)
        
        result = inference_detector(model, img)
        json_generation_float(file_name, img, result)


if __name__ == '__main__':
    main()
