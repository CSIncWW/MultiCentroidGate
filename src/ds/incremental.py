import random
import numpy as np 
from PIL import Image 

import warnings
import torchvision
 
warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)

import torch 

from utils.data_utils import construct_balanced_subset

class IncrementalDataset:
    def __init__(
        self,
        train_dataset,
        test_dataset, 
        val_dataset=None,
        validation_split=0.0, # if val is none
        random_order=None,
        base_classes=10,
        increment=10, #increment number. 
    ):
        # The info about incremental split
        #the number of classes for each step in incremental stage
        self.base_task_size = base_classes
        self.increment = increment
        self.increments = []
        self.random_order = random_order
        self.validation_split = validation_split

        #-------------------------------------
        #Dataset Info
        #-------------------------------------
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset

        self.train_transform = train_dataset.transform
        self.test_transform = test_dataset.transform
        self.val_transform = val_dataset.transform if val_dataset is not None else self.test_transform

        self._setup_data()

        # memory Mt
        self.data_memory = None
        self.targets_memory = None
        self.idx_task_memory = None
        # Incoming data D_t
        self.data_cur = None
        self.targets_cur = None
        self.idx_task_cur = None
        # Available data \tilde{D}_t = D_t \cup M_t
        self.data_inc = None  # Cur task data + memory
        self.targets_inc = None
        self.idx_task_inc = None

        #Current states for Incremental Learning Stage.
        self._current_task = 0
 

    @property
    def n_tasks(self):
        return len(self.increments)

    def new_task(self):
        if self._current_task >= len(self.increments):
            raise Exception("No more tasks.")

        min_class, max_class, x_train, y_train, x_test, y_test = self._get_cur_step_data_for_raw_data()

        self.data_cur, self.targets_cur = x_train, y_train

        if self.data_memory is not None:
            if len(self.data_memory) != 0:
                x_train = np.concatenate((x_train, self.data_memory))
                y_train = np.concatenate((y_train, self.targets_memory))

        self.data_inc, self.targets_inc = x_train, y_train
        self.data_test_inc, self.targets_test_inc = x_test, y_test

        trainset = self.get_custom_dataset("train", "train") # self._get_loader(x_train, y_train, mode="train")
        valset =  self.get_custom_dataset("val", "test") # self._get_loader(x_test, y_test, shuffle=False, mode="test")
        testset =  self.get_custom_dataset("test", "test") # self._get_loader(x_test, y_test, shuffle=False, mode="test")

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "increment": self.increments[self._current_task],
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": len(x_train),
            "n_test_data": len(y_train),
        }

        self._current_task += 1
        return task_info, trainset, valset, testset

    def _get_cur_step_data_for_raw_data(self):
        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])

        x_train, y_train = self._select_range(self.data_train, self.targets_train, low_range=min_class, high_range=max_class)
        x_test, y_test = self._select_range(self.data_test, self.targets_test, low_range=0, high_range=max_class) # yes. 
        return min_class, max_class, x_train, y_train, x_test, y_test

    #--------------------------------
    #           Data Setup
    #--------------------------------
    def _setup_data(self):
        # origin data.
        self.data_train, self.targets_train = [], []
        self.data_test, self.targets_test = [], []
        self.data_val, self.targets_val = [], []
        self.increments = []
        self.class_order = []

        self._split_dataset_task(self.train_dataset, self.val_dataset, self.test_dataset)

        # !list
        self.data_train = np.concatenate(self.data_train)
        self.targets_train = np.concatenate(self.targets_train)
        self.data_val = np.concatenate(self.data_val)
        self.targets_val = np.concatenate(self.targets_val)
        self.data_test = np.concatenate(self.data_test)
        self.targets_test = np.concatenate(self.targets_test)

    def _split_dataset_task(self, train_dataset, val_dataset, test_dataset):
        increment = self.increment

        x_train, y_train = train_dataset.data, np.array(train_dataset.targets)

        if val_dataset is not None:
            x_val, y_val = val_dataset.data, np.array(val_dataset.targets)
        else:
            x_val, y_val, x_train, y_train = self._split_train_val(x_train, y_train, self.validation_split)

        x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

        # Get Class Order
        if self.random_order is not None:
            self.class_order = order = self.random_order
            y_train = self._map_new_class_index(y_train, order)
            y_val = self._map_new_class_index(y_val, order)
            y_test = self._map_new_class_index(y_test, order)
        else:
            self.class_order = np.unique(y_train)

        # self.increments = [increment for _ in range(len(order) // increment)]
        if ((len(order) - self.base_task_size) % increment) != 0:
            print("Warning: not dividible")
        if self.base_task_size == 0:
            print("Warning: base task == 0")
        self.increments = [self.base_task_size] + [increment for _ in range((len(order) - self.base_task_size) // increment)]

        self.data_train.append(x_train)
        self.targets_train.append(y_train)
        self.data_val.append(x_val)
        self.targets_val.append(y_val)
        self.data_test.append(x_test)
        self.targets_test.append(y_test)

    @staticmethod
    def _split_train_val(x, y, validation_split=0.0):
        from sklearn.model_selection import train_test_split
        if validation_split != 0.0:
            x_train, y_train, x_val, y_val = train_test_split(x, y, test_size=validation_split, stratify=True)
        else:
            x_train, y_train, x_val, y_val = x, y, np.array([]), np.array([])
        return x_val, y_val, x_train, y_train

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))

    def _select_range(self, x, y, low_range=0, high_range=0):
        idxes = sorted(np.where(np.logical_and(y >= low_range, y < high_range))[0])
        if isinstance(x, list):
            selected_x = [x[idx] for idx in idxes]
        else:
            selected_x = x[idxes]
        return selected_x, y[idxes]

    def get_custom_dataset(self, data_source="train", transform_type="train", balanced=False, oversample=False):
        """Returns a custom loader.

        :param class_indexes: A list of class indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """

        assert data_source in ["train", "train_cur", "val", "test", "memory"]
        if data_source == "train":
            x, y = self.data_inc, self.targets_inc
        elif data_source == "train_cur":
            x, y = self.data_cur, self.targets_cur
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test_inc, self.targets_test_inc
        elif data_source == "memory":
            x, y = self.data_memory, self.targets_memory
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        assert transform_type in ["train", "test"]
        if transform_type == "train":
            trsf = self.train_transform
        else:
            trsf = self.test_transform

        if balanced:
            x, y = construct_balanced_subset(x, y, oversample)

        return DummyDataset(x, y, trsf) 


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform):
        self.x, self.y = x, y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y, = self.x[idx], self.y[idx]
        if isinstance(x, np.ndarray): # cifar100
            x = Image.fromarray(x)
        else: # Imagnetnet
            with Image.open(x) as f:
                x = f.convert("RGB")
        x = self.transform(x)
        return x, y

class DummyDataset3(torch.utils.data.Dataset):
    def __init__(self, x, y, z, transform):
        self.x, self.y, self.z = x, y, z
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y, z, = self.x[idx], self.y[idx], self.z[idx]
        if isinstance(x, np.ndarray): # cifar100
            x = Image.fromarray(x)
        else: # Imagnetnet
            with Image.open(x) as f:
                x = f.convert("RGB")
        x = self.transform(x)
        return x, y, z

class WeakStrongDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, weak_transform, strong_transform):
        self.x, self.y = x, y
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x, y, = self.x[idx], self.y[idx]
        if isinstance(x, np.ndarray): # cifar100
            x = Image.fromarray(x)
        else: # Imagnetnet
            x = Image.open(x) 
            with Image.open(x) as f:
                x = f.convert("RGB")
        weak_x = self.weak_transform(x)
        strong_x = self.weak_transform(x)
        return weak_x, strong_x, y

torchvision.set_image_backend('accimage')