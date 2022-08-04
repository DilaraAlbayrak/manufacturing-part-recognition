## https://github.com/RBirkeland/MVCNN-PyTorch/blob/master/custom_dataset.py

from torch.utils.data.dataset import Dataset
import os
from PIL import Image

class MultiViewDataSet(Dataset):

    def set_dataset_train_val(self, root, data_type):
        # root / <label>  / <train/test> / <item> / <view>.png
        for label in os.listdir(root): # Label
            for item in os.listdir(root + '/' + label + '/' + data_type):
                views = []
                for view in os.listdir(root + '/' + label + '/' + data_type + '/' + item):
                    views.append(root + '/' + label + '/' + data_type + '/' + item + '/' + view)
                   
                self.x.append(views)
                self.y.append(int(label))

    def set_dataset_test(self, root):
        # root / <label>  / <train/test> / <item> / <view>.png
        views1 = []
        views2 = []
        views3 = []

        for label in sorted(os.listdir(root)): # Label
        
            view_path = root + '/' + label + '/'
            
            for view in sorted(os.listdir(view_path)):
                view_subpath = view_path + view + '/'
                
                for file in sorted(os.listdir(view_subpath)):
                    
                    if "view1" in file:
                        views1.append(view_subpath + file)
                    elif "view2" in file:
                        views2.append(view_subpath + file)
                    elif "view3" in file:
                        views3.append(view_subpath + file)
                    
        for v1,v2,v3 in zip(views1, views2, views3):
            
            views = []
            
            views.append(v1)
            views.append(v2)
            views.append(v3)
            
            self.x.append(views)
            self.y.append(v1.split('/')[1])


    def __init__(self, root, data_type, transform=None, target_transform=None, is_inference=False):
        self.x = []
        self.y = []
        self.root = root


        self.transform = transform
        self.target_transform = target_transform

        if not is_inference:
            self.set_dataset_train_val(root, data_type)
        else:
            self.set_dataset_test(root)


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []

        for view in orginal_views:
            im = Image.open(view)
            im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            views.append(im)

        return views, self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
