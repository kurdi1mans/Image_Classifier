import torch
from torch import nn,optim
from torchvision import transforms,datasets,models
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
import random
import json
from workspace_utils import active_session
from PIL import Image
import numpy as np
from Model_Builder import rebuild_model
from ArgumentParser import get_predict_arguments




def resize_image(image,short_dimension=256):
    target_ratio = 1
    if image.width <= image.height:
        target_ratio = short_dimension/image.width
    else:
        target_ratio = short_dimension/image.height
    return image.resize([int(image.width*target_ratio),int(image.height*target_ratio)])

def crop_center_image(image,width,height):
    w = image.width
    h = image.height

    L = (w-width)/2
    T = (h-height)/2
    R = (w+width)/2
    B = (h+height)/2
    return image.crop((L,T,R,B))

def normalize_color_channels(image,means,deviations):
    image = image/255
    image = (image-means)/deviations
    return image
    
def process_image(image):
    image = resize_image(image,256)
    image = crop_center_image(image,224,224)
    np_image = np.array(image)
    np_image = normalize_color_channels(np_image,[0.485,0.456,0.406],[0.229,0.224,0.225])
    return np_image.transpose(2,0,1)


def predict(image_path, model, topk=5):
    test_image = Image.open(image_path)
    processed_image = process_image(test_image)
    processed_image = torch.from_numpy(processed_image)
    processed_image = processed_image.unsqueeze(0)
    model.eval()
    model.cpu()
    model.double()
    logps = model(processed_image)
    ps = torch.exp(logps)
    top_p,top_classes = ps.topk(topk,dim=1)
    top_p = top_p.squeeze(0).tolist()
    top_classes = top_classes.squeeze(0).tolist()
    top_classes = list(map(model.idx_to_class.get, top_classes))
    
    return top_p,top_classes


def print_predictions(top_p,top_classes,cat_to_name):
    
    for i in range(len(top_classes)):
        print(f'Prediction Category: {top_classes[i]} | Category Name: {cat_to_name[top_classes[i]]} | Prediction Probability: {top_p[i]} ')

def main():
    
    args = get_predict_arguments()
    checkpoint = torch.load(args.checkpoint)

    checkpoint_model = rebuild_model(checkpoint)
    top_p,top_classes = predict(args.image_path, checkpoint_model, topk=args.top_k)
    
     
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    print_predictions(top_p,top_classes,cat_to_name)
    
if __name__ == '__main__':
    main()
    
    
