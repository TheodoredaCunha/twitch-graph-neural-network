"""
Class for Creating the GNN Model
"""

import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
import numpy as np
import os
import pandas as pd
import json
from dataset import TwitchDataset
import configparser

config = configparser.ConfigParser()
config.read('C://Users//heodore da Cunha//Desktop//project//graph neural net//graph-neural-network//config.ini')
print(config)
folder_path = config['dataset']["folder_path"]
