from abc import abstractmethod, ABCMeta
from torch.utils.data import Dataset


class BaseDataset(Dataset, metaclass=ABCMeta):

    @abstractmethod
    def __len__(self):
        pass
