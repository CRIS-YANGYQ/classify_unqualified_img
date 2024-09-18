import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class YangDataset(Dataset):
    """Base class for loading img

    Args:
        img_dir_path(str): The path to the image file dir.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.svg', '.webp', '.ico']

    def __init__(
        self,
        dataseta_df: pd.DataFrame,
        trans: transforms
    ):

        self.df = dataseta_df
        self.transform = trans
    
    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['file']
        label = 1 if row['Class'] == 'Qualified' else 0  # 假设你用1表示Qualified, 0表示Unqualified
        
        try:
            img = cv2.imread(img_path)
            if img is not None:
                # 处理并添加到列表
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
                img_tensor = self.transform(img)
                
        except Exception as e:
            print(f"Error loading image at {img_path}: {e}")
            return None, None
        
        return img_tensor, label
    




