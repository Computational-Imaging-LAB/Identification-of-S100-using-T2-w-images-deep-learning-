import glob
from tqdm import tqdm
import nibabel as nib
import numpy as np
import preprocess
path='/cta/users/abas/Desktop/Embeddings/'
path+='/gliom_data/Meningiom/nii_Meningioma_directory_April2021'
path+='/*/*/Segmentations/*FLAIR_HYP*.ni*'
segmentations=glob.glob(path)
out_path='/cta/users/abas/Desktop/Embeddings/masked_images_MEN_FLAIR/'
shape_problem=[]
prg=preprocess.preGlioma()
for idx,segs in tqdm(enumerate(segmentations),total=len(segmentations)):

    anat_list=segs.split('/')[:-2]
    anat='/'.join(anat_list)
    #anat+='/Anatomic/*T2*DARK*FLUID*/*.nii'
    anat+='/Anatomic/*DARK*FLUID*/*.nii'
    anat_img_path=glob.glob(anat)
    if len(anat_img_path)==0:
        print(anat_img_path)
        continue
    anat_img_nii=nib.load(anat_img_path[-1])
    anat_img=anat_img_nii.get_data()
    anat_img=prg.normalize(anat_img)
    seg_img=nib.load(segs).get_data()

    x,y,z=np.where(seg_img>np.mean(seg_img))
    seg_img[np.min(x):np.max(x),np.min(y):np.max(y),np.min(z):np.max(z)]=1
    if seg_img.shape!=anat_img.shape:
        print(seg_img.shape,anat_img.shape)
        print(segs)
        shape_problem.append(anat_list[-1])
        continue
    masked_im=anat_img*seg_img
    x,y,z=np.where(masked_im>np.mean(masked_im))
    masked_im=masked_im[np.min(x):np.max(x),np.min(y):np.max(y),np.min(z):np.max(z)]
    nib.save( nib.Nifti1Image(masked_im,affine=anat_img_nii.affine),out_path+anat_list[-2]+'_T1_OUT.nii.gz')

    
