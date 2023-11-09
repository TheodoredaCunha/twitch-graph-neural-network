"""
Class for Training the GNN Model
"""

import pandas as pd
import configparser
from dataset import TwitchDataset

config = configparser.ConfigParser()
config.read('config.ini')
folder_path = config["dataset"]["folder_path"]




dataset = TwitchDataset(folder_path)