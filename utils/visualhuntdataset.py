from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torchvision.transforms as T
from transformers import AutoFeatureExtractor, AutoModel

class VisualHuntDataset(Dataset):

    def __init__(self, root_dir, df, nn_arch="TRN"):
        self.df = df
        self.root_dir = root_dir
        self.resize_transform = T.Resize((224, 224))
        self.nn_arch= nn_arch
        self.processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_file = self.root_dir + "/" + self.df["image_id"][idx]
        positive = self.df['pos'][idx]
        negative = self.df['neg'][idx]
        image = Image.open(image_file).convert("RGB")

        encoding={}
        encoding["image_name"] = self.df["image_id"][idx]

        if self.nn_arch == "CNN":
            image = self.resize_transform(image)
            encoding["anchor"]  = transforms.ToTensor()(image)
        else:
            pixel_vals = self.processor(image, return_tensors="pt").pixel_values #The preprocessor will take care of resizing.
            encoding["anchor"] = {"pixel_values" : pixel_vals.squeeze(0)}

        encoding["pos_attr"] = positive
        encoding["neg_attr"] = negative


        return encoding