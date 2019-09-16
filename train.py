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
from Model_Builder import create_model

from ArgumentParser import get_train_arguments




def set_optimizer(model,learning_rate):
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
    return optimizer
def set_criterion():
    return nn.NLLLoss()
def get_accuracy(logps,labels):
    ps = torch.exp(logps)
    top_p,top_class = ps.topk(1,dim=1)
    equals = top_class == labels.view(*top_class.shape)
    batch_accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
    return batch_accuracy
def run_testing(model,criterion,device,validloader,test_type,validation_losses,validation_accuracies):
    model.eval()
    test_loss_aggregate = 0
    accuracy_aggregate = 0
    with torch.no_grad():
        for inputs,labels in validloader:
            inputs,labels = inputs.to(device),labels.to(device)
            logps = model(inputs)
            loss = criterion(logps,labels)
            test_loss_aggregate += loss.item()
            accuracy_aggregate += get_accuracy(logps=logps,labels=labels)
    if validation_losses!=None: validation_losses.append(test_loss_aggregate/len(validloader))
    if validation_accuracies!=None: validation_accuracies.append(accuracy_aggregate/len(validloader))
    print(test_type+f" Loss: {test_loss_aggregate/len(validloader):.3f}")
    print(test_type+f" Accuracy: {accuracy_aggregate/len(validloader):.3f}")

def run_training(model, optimizer, criterion, device, epoch,epochs, test_cycle, trainloader, validloader, train_losses, validation_losses, validation_accuracies):
    # train by batch
    model.train()
    step = 0
    train_loss_aggregate = 0
    for inputs,labels in trainloader:
        step+=1
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps,labels)
        loss.backward()
        optimizer.step()
        train_loss_aggregate += loss.item()
    
        # every couple of batches (determined by test cycle) do testing and print all metrics
        if step % test_cycle == 0:
            #print epoch
            #print step
            print("-------------------------")
            print(f"Epoch: {epoch+1}/{epochs}")
            print(f"Step: {step}")
            #print train loss average
            print(f"Train Loss: {train_loss_aggregate/test_cycle:.3f}")
            train_losses.append(train_loss_aggregate/test_cycle)
            run_testing(
                            model=model,
                            criterion=criterion,
                            device=device,
                            validloader=validloader,
                            test_type="Validation",
                            validation_losses=validation_losses,
                            validation_accuracies=validation_accuracies
                        )
            #reset train loss aggregate
            train_loss_aggregate = 0

def run_epoch(model, optimizer, criterion, device, epoch, epochs, test_cycle, trainloader, validloader, train_losses, validation_losses, validation_accuracies):
    run_training(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                epochs=epochs,
                test_cycle=test_cycle,
                trainloader=trainloader,
                validloader=validloader,
                train_losses=train_losses,
                validation_losses=validation_losses,
                validation_accuracies=validation_accuracies
                )
    
def set_device(gpu=False):
    if gpu:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            raise Exception("GPU is not available; hence, Device cannot be set to GPU.")
    else:
        device = "cpu" 
    return device





def invert_class_to_idx(class_to_idx):
        return dict(map(reversed, class_to_idx.items()))



def save_model_checkpoint(model,path):
    model.to('cpu')
    checkpoint ={
                    'state_dict':model.classifier.state_dict(),
                    'classifier':model.classifier,
                    'class_to_idx':model.class_to_idx,
                    'idx_to_class':model.idx_to_class,
                    'architecture':model.architecture,
                    'hidden_units':model.hidden_units
                }
    torch.save(checkpoint,path)
def save_optimizer_checkpoint(epochs,optimizer,path):
    checkpoint ={
                    'optimizer_dict':optimizer.state_dict(),
                    'epochs':epochs
                }
    torch.save(checkpoint,path)
    


def main():
    args = get_train_arguments()
    device = set_device(args.gpu)
              
    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([
                                            transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                        ])
    test_transforms = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                        ])

    
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=test_transforms)

    
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=64)
    
   
    model = create_model(arch=args.architecture,hidden_units=args.hidden_units)

    
    model.to(device)

    optimizer = set_optimizer(model,learning_rate = args.learning_rate)
    criterion = set_criterion()

    epochs = args.epochs
    test_cycle = 10

    train_losses = []
    validation_losses = []
    validation_accuracies = []


    with active_session():
        for epoch in range(epochs):
            run_epoch(
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=device,
                        epoch=epoch,
                        epochs=epochs,
                        test_cycle=test_cycle,
                        trainloader=train_loader,
                        validloader=valid_loader,
                        train_losses=train_losses,
                        validation_losses=validation_losses,
                        validation_accuracies=validation_accuracies
                    )
    
    model.class_to_idx = train_data.class_to_idx
    model.idx_to_class = invert_class_to_idx(train_data.class_to_idx)
    model.architecture = args.architecture
    model.hidden_units = args.hidden_units
    
    save_model_checkpoint(model,'model_checkpoint.pth')
    save_optimizer_checkpoint(epochs,optimizer,'optimizer_checkpoint.pth')
    
if __name__ == '__main__':
    main()


