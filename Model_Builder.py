from torchvision import models
from torch import nn

def create_model(arch,hidden_units):
    model = None
    if arch == "vgg16":
        model = models.vgg16(pretrained = True)
        in_features = model.classifier[0].in_features
    elif arch == "densenet121":
        model = models.densenet121(pretrained = True)
        in_features = model.classifier.in_features
    else:
        raise Exception("Provided value for parameter --arch is not supported. Execution aborted.")

    # switch off the gradients of the convolutional feature recognition layer
    for param in model.parameters():
        param.requires_grad = False
    
    
    # replace the classifier layer
    # parameters here require gradient by default
    model.classifier = nn.Sequential(
        nn.Linear(in_features,hidden_units[0]),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units[0],hidden_units[1]),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units[1],102),
        nn.LogSoftmax(dim=1)
    )
    return model

def rebuild_model(checkpoint):
    arch = checkpoint['architecture']
    hidden_units = checkpoint['hidden_units']
    
    model = None
    if arch == "vgg16":
        model = models.vgg16(pretrained = True)
        in_features = model.classifier[0].in_features
    elif arch == "densenet121":
        model = models.densenet121(pretrained = True)
        in_features = model.classifier.in_features
    else:
        raise Exception("Provided value for parameter --arch is not supported. Execution aborted.")

    # switch off the gradients of the convolutional feature recognition layer
    for param in model.parameters():
        param.requires_grad = False
    
    
    # replace the classifier layer
    # parameters here require gradient by default
    model.classifier = checkpoint['classifier']
    model.classifier.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.architecture = checkpoint['architecture']
    model.hidden_units = checkpoint['hidden_units']
    
    
    return model