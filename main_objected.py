import pprint
from model_builder import build_models

#from train_dataloader import dataloader
from meningiom_dataloader import dataloader
import torch
import torch.nn as nn

from torchsummary import summary

import pandas as pd
import preprocess
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb
from torchvision.utils import save_image
import json
import os
import torch.nn.functional as F


def count_parameters(model):
    """Counts the number of parameters in a model.

    Args:
        model (torch.model): Model to count parameters

    Returns:
        [int]: Count of all trainable parameters in the model
    """    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


wandb.login()


def train_epoch(network, classifier, loader, val_loader, optimizer, optimizer_classifier, criterion, device: str, slices: int):
    """ Training loop for the proposed model

    Args:
        network (torch.model): torch pre-trained model aka feature extractor (resnet34)
        classifier (torch.model): Classifier model for the extracted features
        loader (dataloader): Dataloader for the training set
        val_loader (dataloader): Dataloader for the validation set
        optimizer (torch.optim.optimizer): Optimizer for the feature extractor
        optimizer_classifier (torch.optimizer): Optimizer for the classifier
        criterion (torch.losses): loss function
        device (str): Device to use for training
        slices (int):  Number of slices to use for the training

    Returns:
        [float]: Cumulative loss for the training set
        [float]: Accuracy for the training set
        [float]: Cumulative loss for the validation set
        [float]: Accuracy for the validation set
        [float]: Accuracy for the training set slice-wise
        [float]: Accuracy for the training set majority voting
        [float]: Accuracy for the training set single-slice positivity
        [int]: Number of class 0
        [int]: Number of class 1
        [int]: Number of class 0 in the evaluation set
        [int]: Number of class 1 in the evaluation set


    """    
        
    

    cumu_loss = 0
    criterion = eval(criterion)
    accuracy_epoch_train = 0
    xc = 0
    xca = 0
    class0 = 0
    class1 = 0
    for _, (datas, targets, features) in tqdm(enumerate(loader), total=len(loader)):
        if len(datas.shape) == 1:
            continue
        datas = datas.permute(3, 0, 1, 2)
        # datas=F.interpolate(datas,scale_factor=2,mode='bilinear',align_corners=True)

        accuracy_train = 0
        xc += 1
        data_sliced, target_sliced = create_the_slices(
            datas, targets, slices=slices, dim=0)
        if targets == 1:
            class1 += 1*datas.shape[0]
        else:
            class0 += 1*datas.shape[0]
        for idx, (data, target) in enumerate(zip(data_sliced, target_sliced)):
            target = torch.tensor([target]*data.shape[0])
            data, target = data.to(
                device), target.to(device)
            optimizer.zero_grad()
            optimizer_classifier.zero_grad()
            network.train()
            network.to(device)
            classifier.to(device)
            xca += 1
            try:
                output = network(data)
                output = classifier(output, torch.cat(
                    features).to(device).float())
            except ValueError:
                continue
            # ➡ Forward pass

            loss = criterion(output, target)
            # ⬅ Backward pass + weight update

            loss.backward()
            optimizer.step()
            optimizer_classifier.step()
            output = torch.softmax(output, dim=1)

            cumu_loss += loss.item()
            accuracy_train += torch.sum((torch.argmax(output,
                                                      dim=1) == target))/len(target)

            wandb.log({"batch loss": loss.item()})
        accuracy_epoch_train += accuracy_train/len(data_sliced)
    cumu_val_loss = 0
    accuracy_epoch = 0

    xn = 0
    xna = 0
    network = network.eval()
    classifier.eval()
    class_eval1 = 0
    class_eval0 = 0
    major_acc = 0
    single_acc = 0

    with torch.no_grad():
        for _, (datas, targets, features) in tqdm(enumerate(val_loader), total=len(val_loader)):
            if len(datas.shape) == 1:
                continue
            datas = datas.permute(3, 0, 1, 2)
            # datas=F.interpolate(datas,scale_factor=2,mode='bilinear',align_corners=True)

            accuracy_train = 0
            data_sliced, target_sliced = create_the_slices(
                datas, targets, slices=slices, dim=0)
            if targets == 1:
                class_eval1 += 1*datas.shape[0]
            else:
                class_eval0 += 1*datas.shape[0]
            xn += 1
            xna += 1
            out_label = 0
            accuracy = 0
            outs = []
            for data, target in zip(data_sliced, target_sliced):
                target = torch.tensor([target]*data.shape[0])
                data, target = data.to(
                    device), target.to(device)
                optimizer.zero_grad()
                network.to(device)
                classifier.to(device)

                output = network(data)
                output = classifier(output, torch.tensor(
                    features).to(device).float())

                # ➡ Forward pass
                loss = criterion(output, target)
                cumu_val_loss += loss.item()
                if torch.sum(target > 0):
                    out_label2 = 0
                out_label += torch.sum(
                    (torch.argmax(torch.softmax(output, dim=1), dim=1)) == target)
                outs += list(torch.argmax(output, dim=1).cpu().numpy())
            outs = np.array(outs)
            single_acc += (np.max(outs)).astype('int') == targets
            major_acc += (np.mean(outs) > 0.5).astype('int') == targets
            accuracy_epoch += out_label/datas.shape[0]
            wandb.log({"batch loss": loss.item(),
                       "batch accuracy": accuracy})

    return cumu_loss / xca, accuracy_epoch/xn, cumu_val_loss / xna, accuracy_epoch_train/xc, single_acc/xn, major_acc/xn, class0, class1, class_eval0, class_eval1


def create_the_slices(image, label, slices: int = 5, dim: int = 0):
    """ Splitting raw input image into slices

    Args:
        image (torch.tensor): Input image
        label (torch.tensor): Target label
        slices (int, optional): Split size. Defaults to 5.
        dim (int, optional): Dimension of splitting. Defaults to 0.

    Returns:
        [float]: list of tensor images,
        [float]: list of tensor labels 
    """    
    images = torch.split(image, slices, dim=dim)
    labels = [label]*len(images)
    return images, labels


def create_model(model_name, classes: int = 2, pretrained: bool = True, requires_grad: bool = True, in_channels: int = 1, custom_pretrained=None, feature_extractor=True):
    """Feature extractor creation

    Args:
        model_name (str): One of [resnetx,densenetx,vggx,inceptionx]
        classes (int, optional): Output of the classes. Defaults to 2. It is unimportant for the model if you use it as feature extractor because we will change the last classifier by IdentityNet()
        pretrained (bool, optional): Pretained flag. Defaults to True.
        requires_grad (bool, optional): Freeze flag. Defaults to True.
        in_channels (int, optional): Input channel(s) of the feature extractor . Defaults to 1.
        custom_pretrained (str, optional): If you have pretrained networks to use to train classifier give the path. Defaults to None.
        feature_extractor (bool, optional): If it is true classes become not important. If no please consider number of classes. Defaults to True.

    Returns:
        [torch.model]: Feature extractor if feature_extractor is true, else model with the out classes and pretrained weights
    """    
    model_cr = build_models(model_name, classes, pretrained,
                            requires_grad, in_channels, custom_pretrained, feature_extractor=feature_extractor)
    if 'resnet' in model_name:
        model = model_cr.build_resnet()
    elif 'efficientnet' in model_name:
        model = model_cr.build_efficientnet()
    elif 'densenet' in model_name:
        model = model_cr.build_densenet()
    elif 'vgg' in model_name:
        model = model_cr.build_vgg()

    return model


def build_optimizer(network, optimizer:str, learning_rate:float):
    """Optimizer creation

    Args:
        network (torch.model): Network to train
        optimizer (str): optimizer type one of [sgd,adam,rmsprop]
        learning_rate (float): float learning rate

    Returns:
        [torch.optim.optimizer]: optimizer
    """    

    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=learning_rate)

    elif optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=learning_rate)
    return optimizer


def build_dataset(input_csv: pd.DataFrame, root_path: str, column='s100', batch_size: int = 1, transform: bool = False, shuffle: bool = False, num_workers: int = 0):
    """Dataset creation

    Args:
        input_csv (pd.DataFrame): Input csv file that contains the names, age, sex, tumor location and targets
        root_path (str): root path for images
        column (str, optional): Column of the target. Defaults to 's100'.
        batch_size (int, optional): It is hard to collate different sized of slices hence use 1 for batch size. Defaults to 1.
        transform (bool, optional): Augmentation flag. Defaults to False.
        shuffle (bool, optional): Shuffle flag. Defaults to False.
        num_workers (int, optional): Count of data workers. Defaults to 0.

    Returns:
        [torch.data.dataloader]: dataloader
    """    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Grayscale(),
        # transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(p=0.35),
        transforms.RandomVerticalFlip(p=0.35),
        # transforms.RandomApply([transforms.ColorJitter(brightness=15,contrast=12,hue=0.2)],p=0.1),
        # transforms.RandomResizedCrop((200,200)),
        #transforms.RandomRotation((-40, 40),fill=(0,)),
        transforms.RandomPerspective(p=0.15, fill=(0,)),
        # transforms.ToTensor(),
        # transforms.RandomCrop(10),

        #transforms.RandomApply(transforms.RandomRotation((-40, 40),fill=(0,)),p=0.25),
        transforms.RandomErasing(p=0.1),
    ])
    # download MNIST training dataset
    if transform:
        datasets = dataloader(input_csv, root_path,
                              column=column, transform=data_transform)
    # ,transform=data_transform)
    else:
        datasets = dataloader(
            input_csv, root_path, column=column)

    loader = torch.utils.data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader


def read_csv(path:str, column: str = 's100'):
    """Read csv file and dropping the nan values of target column

    Args:
        path (str): Path to csv file
        column (str, optional): Column to drop nans. Defaults to 'IDHConsensus'.

    Returns:
        [pandas.DataFrame]: Dataframe that cleaned from nan vals of target column
    """    
    input_csv2 = pd.read_excel(path)
    input_csv2.dropna(subset=[column], inplace=True)
    input_csv2.reset_index(drop=True, inplace=True)
    return input_csv2





sweep_config = {
    'method': 'random'
}

metric = {
    'name': 'accuracy',
    'goal': 'maximize'
}

sweep_config['metric'] = metric

parameters_dict = {
    'optimizer': {
        'values': ['rmsprop', 'sgd'],
    },
    'learning_rate': {'distribution': 'uniform', 'min': 0.00001, 'max': 0.001},
    'patch_size': {'values': [5]},
    'epochs': {'values': [100, 250, 150]},
    'out_classes': {'value': 2},
    'models': {'values': ['resnet18', 'resnet34', 'efficientnet-b2']},
    'path': {'value': '/cta/users/abas/Desktop/Embeddings/meningiom_info_acıbadem_2610.xlsx'},
    'val_path': {'value': '/cta/users/abas/Desktop/Embeddings/validation_data.xlsx'},
    'column': {'value': 'NF2'},
    'root_path': {'value': '/cta/users/abas/Desktop/Embeddings/masked_images_MEN_T2/'},
    'device': {'value': 'cuda:1'},
    'criterion': {'value': 'nn.CrossEntropyLoss(weight=torch.tensor([1,2]).to(\'cuda:1\').float())'},
    'EarlyStopping': {'value': 3},
    'last_epoch': {'value': 0},
    'earlystopped': {'value': False},








}

sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(
    sweep_config, project='Weighted3-NF2-6-2')


ls = []
prg = preprocess.preGlioma()
# nib.load('/cta/users/abas/Desktop/Meningiom/MeningiomData/gliom_data/Gliom/nii_gliom_boun/nii_gliom_directory/G0001/T0001/Segmentations/T0001_T2_HYP.nii')


class classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(classifier, self).__init__()

        self.cls1 = nn.Linear(in_features=in_features,
                              out_features=in_features//2, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.5, inplace=False)
        self.cls2 = nn.Linear(in_features=in_features//2,
                              out_features=in_features//8, bias=True)
        self.cls3 = nn.Linear(in_features=(in_features//8)+2,
                              out_features=out_features, bias=True)

    def forward(self, x, features):

        x = self.cls1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.cls2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.cls3(torch.cat((x, features.repeat(x.shape[0], 1)), dim=1))
        return x


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        csv = read_csv(config.path, config.column)
        train, validate, test = \
            np.split(csv.sample(frac=1, random_state=61),
                     [int(.6*len(csv)), int(.8*len(csv))])

        train.reset_index(drop=True, inplace=True)
        validate.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        loader = build_dataset(
            train, config.root_path, column=config.column, transform=True, shuffle=True)
        val_loader = build_dataset(
            validate, config.root_path, column=config.column)
        test_loader = build_dataset(
            test, config.root_path, column=config.column)

        network = create_model(config.models, config.out_classes,
                               pretrained=True, requires_grad=True, in_channels=1)

        optimizer = build_optimizer(
            network, config.optimizer, config.learning_rate)

        val_loss_past = 99999
        train_loss_past = 0
        best_accuracy = 0
        try:
            classifier_model = classifier(
                list(network.children())[-1].feat, config.out_classes)
        except:
            classifier_model = classifier(
                list(network.children())[-2].feat, config.out_classes)

        optimizer_classifier = build_optimizer(
            classifier_model, config.optimizer, config.learning_rate)
        early_trigger = 0
        for epoch in range(config.epochs):

            avg_loss, accuracy, val_loss, train_acc, single_acc, major_acc, class0, class1, val_class0, val_class1 = train_epoch(
                network, classifier_model, loader, val_loader, optimizer, optimizer_classifier, config.criterion, device=config.device, slices=config.patch_size)
            wandb.log({"loss": avg_loss, "epoch": epoch, "val_loss": val_loss,
                       "accuracy": accuracy, "train_acc": train_acc, "major_Acc": major_acc, "single_Acc": single_acc, "class0": class0, "class1": class1, "val_class0": val_class0, "val_class1": val_class1})
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                early_trigger = 0
                run_name = wandb.run.name
                try:
                    os.mkdir(
                        f'/cta/users/abas/Desktop/Embeddings/optimized/{config.column}_model_{run_name}_{best_accuracy:.4f}')
                except:
                    os.mkdir(
                        f'/cta/users/abas/Desktop/Embeddings/optimized/2_{config.column}_model_{run_name}_{best_accuracy:.4f}')

                checkpoint = {'model': network,
                              'classifier': classifier_model,
                              # 'classifier_state_dict':classifier_model.state_dict(),
                              'state_dict': network.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'epoch': epoch,
                              'Target': config.column,
                              'best_accuracy': best_accuracy,
                              'loaders': loader,
                              'val_loaders': val_loader,
                              'test_loader': test_loader,
                              'test_set': test,
                              'val_set': validate,
                              'train_set': train,
                              }

                torch.save(
                    checkpoint,  f'/cta/users/abas/Desktop/Embeddings/optimized/{config.column}_model_{run_name}_{best_accuracy:.4f}/model_{best_accuracy:.4f}.pth')
                torch.save(classifier_model.state_dict(
                ), f'/cta/users/abas/Desktop/Embeddings/optimized/{config.column}_model_{run_name}_{best_accuracy:.4f}/model_{best_accuracy:.4f}.pt')
                config.update({'last_epoch': {'value': epoch}})
                with open(f'/cta/users/abas/Desktop/Embeddings/optimized/{config.column}_model_{run_name}_{best_accuracy:.4f}/model_config.json', 'w') as fp:
                    json.dump(dict(config), fp)

            if val_loss_past < val_loss and train_loss_past > avg_loss:
                early_trigger += 1
                if early_trigger >= config.EarlyStopping:
                    config.update({'last_epoch': {'value': epoch}})
                    config.update({'earlystopped': {'value': True}})
                    break
            else:
                early_trigger = 0
            val_loss_past = val_loss
            train_loss_past = avg_loss


wandb.agent(sweep_id, train, count=61)
