"""
Class for Loading the Dataset
"""

import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
import numpy as np
import os
import pandas as pd
import json

class TwitchDataset(Dataset):
    def __init__(self, root, transform = None, pre_transform = None):
        super(TwitchDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        Returns the names of the files that stores the data.
        twitch_edges.json stores information about node connections
        twitch_target.csv stores labels (binary)
        """

        return ['twitch_edges.json', 'twitch_target.csv']
    
    @property
    def processed_file_names(self):
        """
        Returns list of names of processed dataset.
        Each sample is saved to its own .pt file
        """
        size = 127094
        return [f'data_{i}.pt' for i in range(size)]
    
    def process(self):
        """
        Converts JSON and CSV to PyTorch's Data class
        Saves data to the processed folder
        """
        f = open(self.raw_paths[0])
        edge_file = json.load(f)
        target_files = pd.read_csv(self.raw_paths[1])

        for i in target_files.index:
            edge_index = edge_file["{}".format(target_files["id"][i])]
            label = target_files["target"][i]

            data = Data(edge_index = edge_index, y = label)
            torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{i}.pt'))
            
    def len(self):
        """
        Returns length/size of the dataset
        """
        return self.data.shape[0]

    def get(self, idx):
        """
        Retrieve a specified index of the dataset
        """
        data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data