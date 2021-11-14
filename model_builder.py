
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
class Identity(nn.Module):
    """Identity network
        This network is used to convert last layer of pre-trained network to the identity if feature_extractor flag is set.
    Args:
        nn ([type]): [description]
    """
    def __init__(self,features):
        super(Identity, self).__init__()
        self.feat=features
        
    def forward(self,y):
        return y


class classifier(nn.Module):
    def __init__(self,in_features,out_features):
        super(classifier, self).__init__()
        self.cls1=nn.Linear(in_features=in_features, out_features=in_features//2, bias=True),
        self.act=nn.ReLU(),
        self.drop=nn.Dropout(p=0.5, inplace=False),
        self.cls2=nn.Linear(in_features=in_features//2, out_features=in_features//4, bias=True),
        self.cls3=nn.Linear(in_features=(in_features//4)+5, out_features=out_features, bias=True)

    def forward(self, x ,features):

        x=self.cls1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.cls2(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.cls3(torch.cat((x,features),dim=1))
        return x



class build_models():


    def __init__(self,model_name, out_classes, pretrained, requires_grad, in_channels,custom_pretrained=None,feature_extractor:bool=False):
        """Inıtilizing the class

        Args:
            model_name (str): One of the name of pretrained classifiers on pytorch
            out_classes (int): Number of classes
            pretrained (bool): Pretraining flag
            requires_grad (bool): Flag to freeze the weights
            in_channels (int): Number of input channels
            custom_pretrained (str, optional):  Defaults to None.
            feature_extractor (bool, optional): Defaults to False.
        """    
        self.model_name = model_name
        self.out_classes = out_classes
        self.pretrained = pretrained
        self.requires_grad = requires_grad
        self.in_channels = in_channels
        self.custom_pretrained = custom_pretrained
        self.feature_extractor = feature_extractor
        

    def build_densenet(self):
        """Generates densenet according the parameters initiliazed

        Returns:
            torch.model: Densenet model
        """        
        model = eval(f'models.{self.model_name}(pretrained={self.pretrained})')
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, self.out_classes)
        model.features.conv0.in_channels = self.in_channels
        if self.in_channels==1:
            model.features.conv0.weight = torch.nn.Parameter(
                torch.mean(model.features.conv0.weight, dim=1).unsqueeze(1))
        for param in model.parameters():
            param.requires_grad = self.requires_grad

        if self.custom_pretrained is not None:
            model.load_state_dict(torch.load(self.custom_pretrained))

        if self.feature_extractor:
            identity = Identity(model.classifier.in_features)
            model.classifier=self.Identity
            return model
        else:
            return model

    def build_resnet(self):
        """Generates resnet according the parameters initiliazed

        Returns:
            torch.model: resnet model
        """        
        model = eval(f'models.{self.model_name}(pretrained={self.pretrained})')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.out_classes)
        model.conv1.in_channels = self.in_channels
        if self.in_channels==1:
            model.conv1.weight = torch.nn.Parameter(
                torch.mean(model.conv1.weight, dim=1).unsqueeze(1))
        for param in model.parameters():
            param.requires_grad = self.requires_grad

        if self.custom_pretrained is not None:
            model.load_state_dict(torch.load(self.custom_pretrained))
        
        if self.feature_extractor:
            identity = Identity(model.fc.in_features)

            model.fc=identity
            return model
        else:
            return model



    def build_efficientnet(self):
        """Generates efficientnet according the parameters initiliazed

        Returns:
            torch.model: efficientnet model
        """        
        
        if self.pretrained:
            model = EfficientNet.from_pretrained(self.model_name, num_classes=self.out_classes)
        else:
            model = EfficientNet.from_name(self.model_name, num_classes=self.out_classes)
        
        
        model._conv_stem.in_channels = self.in_channels

        if self.in_channels==1:
            model._conv_stem.weight = torch.nn.Parameter(
                torch.mean(model._conv_stem.weight, dim=1).unsqueeze(1))

        for param in model.parameters():
            param.requires_grad = self.requires_grad
        
        if self.feature_extractor:
            identity = Identity(model._fc.in_features)
            model._fc=identity
            return model

        if self.custom_pretrained is not None:
            model.load_state_dict(torch.load(self.custom_pretrained))

        return model


    def build_vgg(self):
        """Builds vgg according the parameters initiliazed

        Returns:
            torch.model: vgg model
        """        
        model = eval(f'models.{self.model_name}(pretrained={self.pretrained})')
        
        num_ftrs = model.classifier[-1].in_features
        model.classifier[6].out_features = self.out_classes
        model.classifier[-1] = nn.Linear(num_ftrs, self.out_classes)

        model.features[0].in_channels = self.in_channels
        if self.in_channels==1:
            model.features[0].weight = torch.nn.Parameter(
                torch.mean(model.features[0].weight, dim=1).unsqueeze(1))
        for param in model.parameters():
            param.requires_grad = self.requires_grad

        if self.custom_pretrained is not None:
            model.load_state_dict(torch.load(self.custom_pretrained))


        if self.feature_extractor:
            identity = Identity(model.classifier[0].in_features)
            model.classifier=identity
            return model
        else:
            return model