import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class getDataset(Dataset):
    def __init__(
        self,
        data_name="Beijing",
        df_data=None,
        transform=None,
        tokenizer=None,
        is_test=False,
        mean=None,
        std=None,
        poi_path=None,
        poi_normalization='sum',
    ):
        super().__init__()
        self.transform = transform
        self.tokenizer = tokenizer
        self.poi_normalization = poi_normalization

        if mean is None:
            mean = [0, 0, 0]
        if std is None:
            std = [1, 1, 1]

        self.text_tensors, self.img_tensors, self.y = [], [], []
        self.region_labels, self.poi_features = [], []

        self.poi_df, self.poi_columns = self._load_poi(poi_path)

        for _, row in df_data.iterrows():
            filename = row["filename"]
            region = filename.replace(".tif.tif", "")
            poi_feature = self._get_poi_feature(region)
            self.poi_features.append(poi_feature)

            img_path = self._get_image_path(data_name, filename)
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            self.img_tensors.append(image)

            text = self._get_text(data_name, filename)
            tokens = self.tokenizer(text)
            self.text_tensors.append(tokens)

            if is_test:
                self.y.append([0.0, 0.0, 0.0])
            else:
                C = (row["c_value"] - mean[0]) / std[0]
                pop = (row["pop_value"] - mean[1]) / std[1]
                light = (row["light_value"] - mean[2]) / std[2]
                self.y.append([C, pop, light])

            self.region_labels.append(region)

        self.unique_regions = sorted(set(self.region_labels))
        self.region_to_id = {r: i for i, r in enumerate(self.unique_regions)}
        self.region_ids = torch.tensor([self.region_to_id[r] for r in self.region_labels], dtype=torch.long)

        if len(self.poi_features) > 0:
            self.poi_features = np.stack(self.poi_features)
        else:
            self.poi_features = np.array([])

    def _load_poi(self, poi_path):
        if poi_path is None or not os.path.exists(poi_path):
            return None, []

        poi_df = pd.read_csv(poi_path)
        poi_columns = [c for c in poi_df.columns if c != 'name']

        if self.poi_normalization == 'sum':
            poi_df[poi_columns] = poi_df[poi_columns].apply(lambda x: x / (x.sum() + 1e-6), axis=1)
        elif self.poi_normalization == 'max':
            poi_df[poi_columns] = poi_df[poi_columns].apply(lambda x: x / (x.max() + 1e-6), axis=1)
        elif self.poi_normalization == 'log':
            poi_df[poi_columns] = np.log1p(poi_df[poi_columns])
        return poi_df, poi_columns

    def _get_poi_feature(self, region):
        if self.poi_df is None:
            return np.array([], dtype=np.float32)
        poi_row = self.poi_df[self.poi_df['name'] == region + '.csv']
        if not poi_row.empty:
            return poi_row[self.poi_columns].values[0].astype(np.float32)
        return np.zeros(len(self.poi_columns), dtype=np.float32)

    def _get_image_path(self, data_name, filename):
        base = {
            "Beijing": "data/beijing_sat/",
            "Shanghai": "data/shanghai_sat/",
            "NewYork": "data/NewYork_file/",
        }
        return os.path.join(base[data_name], filename.replace(".tif.tif", ".tif"))

    def _get_text(self, data_name, filename):
        if data_name == "NewYork":
            text_path = os.path.join("data/NewYork_generated_descriptions_Summary",
                                     filename.replace(".tif.tif", "_description.txt"))
        else:
            folder = "beijing_descriptions" if data_name == "Beijing" else "Shanghai_descriptions"
            text_path = os.path.join(f"data/{folder}/",
                                     filename.replace(".tif.tif", "_description_description.txt"))
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        poi_feature = torch.from_numpy(self.poi_features[index]) if len(self.poi_features) > 0 else torch.tensor([])
        return (
            self.img_tensors[index],
            self.text_tensors[index],
            self.region_ids[index],
            np.float32(self.y[index]),
            poi_feature.float(),
        )

    def get_poi_feature_dim(self):
        return len(self.poi_columns) if self.poi_columns else 0