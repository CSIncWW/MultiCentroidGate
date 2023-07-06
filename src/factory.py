import torch
from torch import nn
from torch import optim

from torchvision.datasets import CIFAR100, CIFAR10, MNIST, SVHN
from ds.dataset_transform import dataset_transform
from ds.dataset_order import dataset_order
from ds.incremental import IncrementalDataset
from timm.data import create_transform
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import importlib
from timm.data.distributed_sampler import RepeatAugSampler
import torch.distributed as dist
import torch.utils.data
from torch.utils.data import DataLoader, WeightedRandomSampler
from ds.sampler import DistributedWeightedSampler, MPerClassKPerTaskSampler, MPerClassSampler, MaxKclassSampler, StratifySampler
from ds.dataset import ImageNet100, ImageNet1K, SVHNWrapper, ImageNet50
from timm.data import Mixup
from ds.incremental import WeakStrongDataset
from timm.data.auto_augment import RandAugment, AugmentOp, rand_augment_ops

from utils.incremental_utils import target_to_task

def create_convnet(cfg):
    convnet_type = cfg['convnet']
    if convnet_type == "resnet18_cifar":
        # from models.backbones.cifar.resnet import ResNet18
        from models.backbones.small.resnet import resnet18
        return resnet18(False), 512 
    elif convnet_type == "resnet32":
        from models.backbones.resnet_cifar import resnet32
        return resnet32(), 64
        # from models.backbones.resnet_cifar2 import resnet32
        # return resnet32(), 64
    elif convnet_type == "preact_resnet":
        from models.backbones.preact_resnet import PreActResNet18
        return PreActResNet18(), 512
    elif convnet_type == "resnet18":
        from models.backbones.resnet import resnet18
        # from torchvision.models.resnet import resnet18
        return resnet18(False), 512
    elif convnet_type == "resnet34":
        from models.backbones.resnet import resnet34
        return resnet34(False), 512
    elif convnet_type == "resnet18_podnet":
        from models.backbones.resnet_podnet import resnet18
        return resnet18(False), 512
    elif convnet_type == "resnet18_c":
        from models.backbones.resnet_c import resnet18
        return resnet18(False), 512
    elif convnet_type == "wide_resnet":
        from models.backbones.wide_resnet import Wide_ResNet
        return Wide_ResNet(16, 8, 0.3, 10), 64 * 8 
    elif convnet_type == "CNN":
        from models.backbones.cnn import CNN
        return CNN(**cfg['args']),  2048 * 4
    elif convnet_type == "simpleCNN":
        from models.backbones.simple_cnn import SimpleCNN
        return SimpleCNN(), 84
    elif convnet_type == "simpleCNN2":
        from models.backbones.simple_cnn2 import SimpleCNN2
        return SimpleCNN2(), 84
    else:
        raise NotImplementedError("Unknwon convnet type {}.".format(convnet_type))

def create_network(cfg): 
    lib = importlib.import_module(f"models.{cfg['network']}") 
    return lib.Model(cfg) 

def create_trainer(cfg, inc_dataset:IncrementalDataset):
    lib = importlib.import_module(f"trainer.{cfg.trainer}")
    return lib.IncModel(cfg, inc_dataset)

def create_data(cfg):
    # cfg['data_folder'] = '~/dataset'
    if cfg['dataset'] == 'CIFAR100':
        train_transform = dataset_transform['CIFAR100']['train']
        test_transform = dataset_transform['CIFAR100']['test']
        trainset = CIFAR100(cfg['data_folder'],
                            train=True,
                            transform=train_transform, download=True)
        testset = CIFAR100(cfg['data_folder'],
                            train=False,
                            transform=test_transform, download=True)
        order = dataset_order['CIFAR100'][cfg.class_order_idx]
    elif cfg['dataset'] == 'ImageNet100':
        train_transform = dataset_transform['ImageNet100']['train']
        test_transform = dataset_transform['ImageNet100']['test']
        trainset = ImageNet100(cfg['data_folder'],
                            meta_root="imagenet_split",
                            train=True,
                            transform=train_transform)
        testset = ImageNet100(cfg['data_folder'],
                            meta_root="imagenet_split",
                            train=False,
                            transform=test_transform)
        order = dataset_order['ImageNet100'][0]
    elif cfg['dataset'] == "ImageNet1K":
        train_transform = dataset_transform['ImageNet1K']['train']
        test_transform = dataset_transform['ImageNet1K']['test']
        trainset = ImageNet1K(cfg['data_folder'],
                            meta_root="imagenet_split",
                            train=True,
                            transform=train_transform)
        testset = ImageNet1K(cfg['data_folder'],
                            meta_root="imagenet_split",
                            train=False,
                            transform=test_transform)
        order = dataset_order['ImageNet1K'][0]
    print(train_transform, test_transform)
    print(f"dataset order: {order}")
    
    return IncrementalDataset(trainset, testset, None, 0, order, cfg.base_classes, cfg.increment)

def create_sampler(cfg, dataset, ddp, shuffle, sampler_type=""):
    if ddp:  # args.distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        if sampler_type == "weight":
            sampler_train = DistributedWeightedSampler(
                dataset, num_replicas=num_tasks, rank=global_rank
            )
        elif sampler_type == "pk":
            pass
        elif sampler_type == "maxk":
            sampler_train = MaxKclassSampler(dataset.y, target_to_task(dataset.y, cfg.increments), cfg.md_k, cfg, global_rank)
        elif sampler_type == "stratify":
            sampler_train = StratifySampler(dataset.y, rank=global_rank)
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
            ) 
    else:
        if shuffle:
            sampler_train = torch.utils.data.RandomSampler(dataset)
        else:
            sampler_train = torch.utils.data.SequentialSampler(dataset)
    return sampler_train

def create_dataloader(cfg, dataset, ddp, shuffle, drop_last=False, sampler_type=""): # [pk, stratify, weight]
    train_sampler = create_sampler(cfg, dataset, ddp, shuffle, sampler_type=sampler_type)
    if not isinstance(train_sampler, StratifySampler):
        batch_size = cfg.batch_size 
    else:
        b = train_sampler.get_batch_size()
        batch_size = int((cfg.batch_size // b + 1) * b)
    return DataLoader(dataset,
                    batch_size,
                    sampler=train_sampler,
                    drop_last=drop_last,
                    num_workers=cfg.num_workers,
                    pin_memory=cfg.pin_mem,
                    generator=torch.Generator(), # so that Dataloader don't change global random state. Same dataloder cross device.
                    )
 
_mean = {
    "ImageNet1K": IMAGENET_DEFAULT_MEAN,
    "ImageNet100": IMAGENET_DEFAULT_MEAN,
    "ImageNet50": IMAGENET_DEFAULT_MEAN,
    "CIFAR100": (0.5071, 0.4867, 0.4408),
    "CIFAR10": (0.5071, 0.4867, 0.4408),
    "MNIST": (0.1307,),
    "SVHN": (0.5, 0.5, 0.5) 
}
_std = {
    "ImageNet1K": IMAGENET_DEFAULT_STD,
    "ImageNet100": IMAGENET_DEFAULT_STD,
    "ImageNet50": IMAGENET_DEFAULT_STD,
    "CIFAR100": (0.5071, 0.4867, 0.4408),
    "CIFAR10": (0.5071, 0.4867, 0.4408),
    "MNIST": (0.1307,),
    "SVHN": (0.5, 0.5, 0.5) 
}

from torchvision.transforms import InterpolationMode
_crop = {
    "ImageNet1K": transforms.RandomResizedCrop((224, 224), interpolation=InterpolationMode.BICUBIC),
    "ImageNet100": transforms.RandomResizedCrop((224, 224), interpolation=InterpolationMode.BICUBIC),
    "ImageNet50": transforms.RandomCrop(32, padding=4),
    "CIFAR100": transforms.RandomCrop(32, padding=4),
    "CIFAR10": transforms.RandomCrop(32, padding=4),
    "MNIST": transforms.CenterCrop(28),
    "SVHN": transforms.RandomCrop(32, padding=4)
}

def a_transform(new_transform, cfg): 
    crop_fn = _crop[cfg.dataset]
    mean=_mean[cfg.dataset]
    std=_std[cfg.dataset] 
    if new_transform == "randaugment":  
        hparams = {
            "translate_const": 100, 
            "magnitude_std": 0.5,
        }
        trans_name = [
            "Rotate",
            "ShearX",
            "ShearY",
            "TranslateXRel",
            "TranslateYRel",
            "AutoContrast",
            "SharpnessIncreasing",
            "Invert", 
            # non.
            "ContrastIncreasing",
            "ColorIncreasing",
            "BrightnessIncreasing",
            # negative 
            "Equalize", 
            "SolarizeIncreasing",
            "SolarizeAdd", 
            "PosterizeIncreasing",
        ]  

        return transforms.Compose([
            crop_fn,
            transforms.RandomHorizontalFlip(0.5),
            RandAugment(rand_augment_ops(cfg.aa_m, hparams=hparams, transforms=trans_name), num_layers=cfg.aa_n),
            transforms.ToTensor(),
            transforms.Normalize(mean, std), 
        ])

def replace_loader_transforms(new_transform, cfg, train_loader):
    train_loader.dataset.transform = a_transform(new_transform, cfg) 
    print(train_loader.dataset.transform)
    return train_loader