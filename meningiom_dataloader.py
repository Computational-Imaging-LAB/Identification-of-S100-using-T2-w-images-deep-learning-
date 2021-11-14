
import glob
import torch
import nibabel as nib
from matplotlib import pyplot as plt
from preprocess import preGlioma

class dataloader():


    def __init__(self, data_csv,root_path,column='NF2',modality='T1',transform=None): 
        self.datas = data_csv
        self.transform=transform
        self.modality=modality  #T2 or T1
        self.root_path=root_path
        self.column=column
        self.prg=preGlioma()

    def __len__(self):
        return min(299,len(self.datas))


    def __getitem__(self, index):
        


        #label=int(self.datas.IDHConsensus[index])
        try:
            label=int(self.datas[self.column][index])            
            #label=int(self.datas.NF2[index])
            img_path=self.root_path+self.datas.AnonymizedName[index]+'*'

        except :
            img_path='M0024*'
        try:
            img_path=glob.glob(img_path)[0]
            img=nib.load(img_path).get_data().astype('float32')  
        except:
            #print(self.datas.AnonymizedName[index],'index:', index)
            return 61,61,61
        #tumoruous_slices=torch.unique(torch.where(seg > 0)[3])
        #img=img[:,:,tumoruous_slices]
        #seg=seg[:,:,tumoruous_slices]

        #img=img*seg
      
        img=self.prg.normalize(img,'min-max')
        if self.transform is not None:
            for i in range(img.shape[-1]):
                img[:,:,i]=self.transform(img[:,:,i])
        
        return img,label,[self.datas['cinsiyet'][index],self.datas['age']['index'],self.datas['Calvarial']['index'],self.datas['Skull'][index]]

    """
    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(520, 520))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(512, 512))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask
        
        """
        
        