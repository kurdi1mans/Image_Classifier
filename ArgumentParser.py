import argparse
from Checker import check
import os

def parse_train_arguments():
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument(action="store",dest="data_directory",help="Root directory where training data is stored.")
    parser.add_argument("--save_dir",action="store",default=".",dest="save_directroy",help="Directory to which the checkpoint after training is stored.")
    parser.add_argument("--save_file",action="store",default="checkpoint.pth",dest="save_file",help="File name to which the checkpoint after training is stored.")
    parser.add_argument("--arch",action="store",default="vgg16",dest="architecture",help="Choose the feature recognition deep net (e.g. vgg13 densenet121)")
    parser.add_argument("--learning_rate",action="store",default=0.001,dest="learning_rate",help="learning rate for the classifier training",type=float)
    parser.add_argument("--hidden_units",nargs="+",default=[500,200],action="store",dest="hidden_units",type=int)
    parser.add_argument("--epochs",default=5,action="store",dest="epochs",type=int)
    parser.add_argument("--gpu",action="store_true",default=False,dest="gpu")
    #print(parser.parse_args())
    return parser.parse_args()

def validate_train_arguments(args):
    check(os.path.isdir(args.data_directory),"Argument for <data_directory> is invalid. No such directory found.")
    check(os.path.isdir(args.save_directroy),"Argument for <save_directroy> is invalid. No such directory found.")
    check(args.learning_rate>0,"Argument for <learning_rate> is invalid. Learning Rate cannot be negative.")
    check(args.epochs>0,"Argument for <epochs> is invalid. Number of Epochs cannot be negative.")
    check(args.architecture in ['vgg16','densenet121'],"Argument for --arch is invalid. This architecture is not supported.")
    check(len(args.hidden_units) == 2,"Argument for --hidden_units is invalid. Only two hidden layers are supported.")

    
    
    
def parse_predict_arguments():
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument(action="store",dest="image_path",help="Path to input image for which a categorization is requested.")
    parser.add_argument(action="store",dest="checkpoint",help="Path to checkpoint at which the predictive model is stored.")
    parser.add_argument("--top_k",default=3,action="store",dest="top_k",type=int)
    parser.add_argument("--category_names",action="store",default="./cat_to_name.json",dest="category_names",help="The path to the JSON file containing the names of the different categories mapped to the numbers given to them.")
    parser.add_argument("--gpu",action="store_true",default=False,dest="gpu")
    return parser.parse_args()

def validate_predict_arguments(args):
    check(os.path.isfile(args.image_path),"Argument for <image_path> is invalid. No such file found.")
    check(os.path.isfile(args.checkpoint),"Argument for <checkpoint> is invalid. No such file found.")
    check(args.top_k>0,"Argument for --top_k is invalid. top_k value cannot be negative.")
    check(os.path.isfile(args.category_names),"Argument for --category_names is invalid. No such file found.")
    
    
    
    
    
def get_train_arguments():
    args = parse_train_arguments()
    validate_train_arguments(args)
    return args

def get_predict_arguments():
    args = parse_predict_arguments()
    validate_predict_arguments(args)
    return args
