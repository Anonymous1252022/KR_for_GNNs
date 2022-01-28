from torch_geometric.data.data import Data
from typing import List


class Dataset:
    def __init__(self, data_list:List[Data], num_features:int, num_classes:int, task_type:str):
        self.__list = data_list
        self.__num_features = num_features
        self.__num_classes = num_classes
        assert task_type in ['s', 'm'], "task_type should be one of {'s', 'm'}"
        self.__task_type = task_type # {s or m}

    def __len__(self) -> int:
        return len(self.__list)

    def __getitem__(self, idx:int) -> Data:
        return self.__list[idx]

    @property
    def num_features(self) -> int:
        return self.__num_features

    @property
    def num_node_features(self) -> int:
        return self.__num_features

    @property
    def num_classes(self) -> int:
        return self.__num_classes

    @property
    def task_type(self) -> str:
        return self.__task_type