import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import torchviz

class classifier(nn.Module):
    def __init__(self,in_features,out_features,features):
        super(classifier, self).__init__()
        
        self.cls1=nn.Linear(in_features=in_features, out_features=in_features//2, bias=True)
        self.act=nn.ReLU()
        self.drop=nn.Dropout(p=0.5, inplace=False)
        self.cls2=nn.Linear(in_features=in_features//2, out_features=in_features//4, bias=True)
        self.cls3=nn.Linear(in_features=(in_features//4)+4, out_features=out_features, bias=True)
        self.features=features
    def forward(self, x ,features=None):
        if features is  None:
            features=self.features
        x=self.cls1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.cls2(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.cls3(torch.cat((x,features.repeat(x.shape[0],1)),dim=1))
        return x




def plot_grid(images,targets,target):

    fig = plt.figure(figsize=(16., 16.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(len(images)//2, 1+(len(images)//(len(images)//2))),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    for idx,(ax, im) in enumerate(zip(grid,images)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im,cmap='gray')
    
        ax.set_title(f'{targets[idx]}-real-{target}', fontsize=15, color= 'blue', fontweight='bold')
        ax.axis('off')
        

    plt.show()

root_path2='/cta/users/abas/Desktop/Embeddings/optimized/model_leafy-sweep-5_0.7374/'
root_path='/cta/users/abas/Desktop/Embeddings/optimized/NF2_model_happy-sweep-13_0.6484/model_0.6484.pth'
root_path3='/cta/users/abas/Desktop/Embeddings/optimized/NF2_model_fresh-sweep-5_0.7112/model_0.7112.pth'
checkpoint=torch.load(root_path, map_location=torch.device('cuda:0'))
#checkpoint2=torch.load(root_path2+'model_0.7374.pth', map_location=torch.device('cuda:0'))
#checkpoint3=torch.load(root_path3, map_location=torch.device('cuda:0'))

root_path='/cta/users/abas/Desktop/Embeddings/optimized/s100_model_sweepy-sweep-7_0.7391/'
#root_path2='/cta/users/abas/Desktop/Embeddings/optimized/s100_model_zesty-sweep-8_0.7283/'
#root_path3='/cta/users/abas/Desktop/Embeddings/optimized/s100_model_ethereal-sweep-5_0.7391/'
#checkpoint=torch.load(root_path+'model_0.7391.pth', map_location=torch.device('cuda:0'))
#checkpoint2=torch.load(root_path2+'model_0.7283.pth', map_location=torch.device('cuda:0'))
#checkpoint3=torch.load(root_path3+'model_0.7391.pth', map_location=torch.device('cuda:1'))





model=checkpoint['model']
classifier_model=checkpoint['classifier']
#model2=checkpoint2['model']
#model3=checkpoint3['model']
#classifier_model3=checkpoint3['classifier']

val_loader=checkpoint['test_loader']
val_loader.transform=None
iteration=iter(val_loader)
torchviz_test=False
a=torch.tensor([1,1,2,3])
if torchviz_test:
    model = nn.Sequential()
    model.add_module('cls1', nn.Linear(512, 256))
    model.add_module('activation', nn.ReLU())
    model.add_module('dropout', nn.Dropout(p=0.5))
    model.add_module('cls2', nn.Linear(256, 128))
    model.add_module('activation', nn.ReLU())
    model.add_module('dropout', nn.Dropout(p=0.5))
    model.add_module('cls3', nn.Linear(128, 2))

    x = torch.randn(1, 512)
    y = model(x)

    dot=torchviz.make_dot(y.mean(), params=dict(model.named_parameters()))
    dot.format = 'svg'
    dot.render()
hld=False
if hld:
    import hiddenlayer as hl

    transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.
    cl2=classifier(in_features=512,out_features=2,features=torch.tensor([0,1,1,2]).float())
    graph = hl.build_graph(cl2, torch.randn(1, 512), transforms=transforms)
    graph.theme = hl.graph.THEMES['blue'].copy()
    graph.save('rnn_hiddenlayer', format='png')

#cl2=classifier(in_features=512,out_features=2,features=torch.tensor([0,1,1,2]).float())
#torch.onnx.export(cl2, torch.randn(1,512), 's100.onnx', input_names='emb', output_names='S100 Expression (0-1)')





for param in model.parameters():
    param.requires_grad = False




with torch.no_grad():
    accuracy=0
    x=0
    class0=0
    class1=0
    accuracy_class=[0,0]
    for data,target,features in val_loader:
        if len(data.shape)==1:
            print('Data shape error')
        else:
            x+=1
            model.eval()
            classifier_model.eval()
            #model2.eval()
            #model3.eval()
            #print(target)
            output=model(data.permute(3,0,1,2).to('cuda:0'))
            output=classifier_model(output,torch.tensor(features).to('cuda:0').float())
            #output2=model2(data.permute(3,0,1,2).to('cuda:0'))
            #"output3=model3(data.permute(3,0,1,2).to('cuda:0'))

            #output2=model3(data.permute(3,0,1,2).to('cuda:0'))
            #output2=classifier_model3(output2,torch.tensor(features).to('cuda:0').float())
            #print(torch.softmax(output,dim=1))
            #print(torch.softmax(output2,dim=1))
            grid_data=data[0,:,:,:]
            grid_data=torch.split(grid_data,1,dim=-1)
            #plot_grid(grid_data,torch.argmax(output,dim=1),int(target))
            
            #result=torch.sum(torch.softmax(output,dim=1),dim=0)+torch.sum(torch.softmax(output2,dim=1),dim=0)
            #result+=torch.sum(torch.softmax(output3,dim=1),dim=0)
            result=torch.sum(torch.argmax(output,dim=1))/output.shape[1]
            #result+=torch.sum(torch.argmax(output2,dim=1))/output.shape[1]
            #result=torch.sum(torch.argmax(output3,dim=1))/output.shape[1]
            #print(result,target)
            if target==1:
                class1+=1
            else:
                class0+=1
            #accuracy+=torch.argmax(torch.softmax(output,dim=0).cpu())==target
            #accuracy+=(result.cpu()>0.5)==target
            outs2=1
            outs=list(torch.argmax(output,dim=1).cpu().numpy())
            #outs2=list(torch.argmax(output2,dim=1).cpu().numpy())
           
            accuracy+=(np.mean(outs)>0.3).astype('int')==target
            if (np.mean(outs)>0.3).astype('int')==target:
                accuracy_class[target]+=1
            print(target,(np.mean(outs)),(np.mean(outs2)))

    print(accuracy,class1,class0)