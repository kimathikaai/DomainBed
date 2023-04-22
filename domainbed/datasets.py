# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from typing import List, Tuple, Dict, Callable, Optional

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW"
]

OVERLAP_TYPES = ["none", "low", "mid", "high", "full", "0", "33", "66", "100"]
SPECIAL_OVERLAP_TYPES = ["0", "33", "66", "100"]

def get_domain_classes(N_c, N_oc, repeat, N_s, seed):
    N_noc = N_c - N_oc
    Q = []
    C = list(range(N_c))

    random_state = np.random.RandomState(seed)

    # choose non overlapping classes
    C_noc = list(random_state.choice(C, replace=False, size=N_noc))
    C_oc = [x for x in C if x not in C_noc]

    # add to queue
    Q.extend(C_noc + list(np.repeat(C_oc, repeat)))

    # Round-robing distribution of classes
    domain_classes = [Q[i::N_s] for i in range(N_s)]

    # assert overlapping classes
    overlap = np.zeros(N_c)
    for cls_list in domain_classes:
        np.add.at(overlap, cls_list, 1)

    assert C_oc == list(np.where(overlap > 1)[0])

    # output
    print("C_noc", C_noc)
    print("C_oc", C_oc)
    print("Q", Q)
    print("domain_classes", domain_classes)

    return domain_classes

class DomainBedImageFolder(ImageFolder):
    """
    Custom class to allow class filtering
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        remove_classes: List[int] = [],
    ):
        super().__init__(root, transform, target_transform)

        # Remove specified classes
        old_samples = self.samples
        self.samples = []
        self.targets = []
        for sample in old_samples:
            _, target = sample
            
            if target not in remove_classes:
                self.samples.append(sample)
                self.targets.append(target)

        self.imgs = self.samples

    def __len__(self) -> int:
        return len(self.samples)

def get_overlapping_classes(class_split: List[List[int]], num_classes: int) -> List[int]:
    """ 
    Return the classes in multiple domains.
    """
    overlap = np.zeros(num_classes)
    for data in class_split:
        np.add.at(overlap, data, 1)

    overlapping_classes = list(np.where(overlap>1)[0])

    return overlapping_classes

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes, test_envs: List[int], domain_class_filter: List[List[int]]):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        assert len(test_envs) == 1, "Not performing leave-one-domain-out validation"
        num_envs = len(environments) 

        self.num_classes = num_classes
        self.overlapping_classes = get_overlapping_classes(domain_class_filter, self.num_classes)

        # Dynamically associate a filter with a domain except for test_envs[0]
        num_filters = len(domain_class_filter)
        assert num_envs-1 == num_filters # b/c exempt first test env
        shift_filter = list(range(num_filters)) + list(range(num_filters))
        shift_filter = shift_filter[test_envs[0]: test_envs[0] + num_filters]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams, class_overlap_id: int):

        self.class_overlap = {
            0: [[0], [1]],
            66: [[0,1], [1]],
            100: [[0,1], [0,1]],
        }
        self.class_overlap_id = class_overlap_id

        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        dataset = TensorDataset(x,y)

        return dataset

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams, domain_class_filter: List[List[int]] = None):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)
        num_envs = len(environments) 

        assert len(test_envs) == 1, "Not performing leave-one-domain-out validation"

        self.idx_to_class = self.get_idx_to_class(os.path.join(root, environments[test_envs[0]]))
        self.num_classes = len(self.idx_to_class)

        if domain_class_filter is None:
            domain_class_filter = [list(range(self.num_classes)) for _ in range(num_envs-1)]

        self.overlapping_classes = get_overlapping_classes(domain_class_filter, self.num_classes)

        # Dynamically associate a filter with a domain except for test_envs[0]
        num_filters = len(domain_class_filter)
        assert num_envs-1 == num_filters # b/c exempt first test env
        shift_filter = list(range(num_filters)) + list(range(num_filters))
        shift_filter = shift_filter[test_envs[0]: test_envs[0] + num_filters]

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            path = os.path.join(root, environment)

            # setup augmentation
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            # setup class filtering
            if i not in test_envs:
                filter = domain_class_filter[shift_filter.pop()]
                all_classes = set(list(self.idx_to_class.keys())) 
                remove_classes = list(all_classes - set(filter))

                env_dataset = DomainBedImageFolder(
                    path,
                    transform=env_transform, 
                    remove_classes=remove_classes)

                env_dataset.is_test_env = False
                env_dataset.allowed_classes = filter
                env_dataset.remove_classes = remove_classes
            else:
                env_dataset = DomainBedImageFolder(
                    path,
                    transform=env_transform)

                env_dataset.is_test_env = True
                env_dataset.allowed_classes = list(range(self.num_classes))
                env_dataset.remove_classes = []

            env_dataset.env_name = environment
            # print(f"\n[info] environment: {env_dataset.env_name}, classes: {env_dataset.allowed_classes}, is_test: {env_dataset.is_test_env}")
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        assert self.num_classes == len(self.datasets[-1].classes)

    def get_overlapping_classes(self, class_split: List[List[int]], num_classes: int) -> List[int]:
        """ 
        Return the classes in multiple domains.
        """
        overlap = np.zeros(num_classes)
        for data in class_split:
            np.add.at(overlap, data, 1)

        overlapping_classes = list(np.where(overlap>1)[0])

        return overlapping_classes

    def get_idx_to_class(self, data_dir: str) -> Dict[int, str]:
        dataset = ImageFolder(data_dir)
        idx_to_class = {}
        for key, value in dataset.class_to_idx.items():
            idx_to_class.update({value: key})

        assert len(dataset.class_to_idx) == len(idx_to_class), "Class and labels are not one-to-one"

        return idx_to_class

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams, overlap, overlap_seed):
        # print(f"[info] {type(self)}, test_envs: {test_envs}, overlap: {class_overlap_id}")
        self.dir = os.path.join(root, "VLCS/")
        num_source_domains = 3
        num_classes = 5
        overlap_config = {
            "none": {"N_oc": 0, "repeat": num_source_domains - 1},
            "low": {"N_oc": 2, "repeat": num_source_domains - 1},
            "mid": {"N_oc": 3, "repeat": num_source_domains - 1},
            "high": {"N_oc": 4, "repeat": num_source_domains - 1},
            "full": {"N_oc": num_classes, "repeat": num_source_domains},
        }

        special_class_overlap = {
            "0": [[0, 1], [2, 3], [4]],
            "33": [[0, 1, 2], [2, 3], [3, 4]],
            "66": [[0, 1, 2], [2, 3, 4], [3, 4, 0]],
            "100": [list(range(5)), list(range(5)), list(range(5))],
        }

        if overlap not in SPECIAL_OVERLAP_TYPES:
            domain_classes = get_domain_classes(
                N_c = num_classes,
                N_oc = overlap_config[overlap]["N_oc"],
                repeat = overlap_config[overlap]["repeat"],
                N_s = num_source_domains,
                seed = overlap_seed
            )
        else:
            domain_classes = special_class_overlap[overlap]

        super().__init__(self.dir, test_envs, hparams['data_augmentation'], 
                         hparams, domain_classes)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    # overlap_type
    def __init__(self, root, test_envs, hparams, overlap, overlap_seed):
        # print(f"[info] {type(self)}, test_envs: {test_envs}, overlap: {class_overlap_id}")
        self.dir = os.path.join(root, "PACS/")
        num_source_domains = 3
        num_classes = 7
        overlap_config = {
            "none": {"N_oc": 0, "repeat": num_source_domains - 1},
            "low": {"N_oc": 3, "repeat": num_source_domains - 1},
            "mid": {"N_oc": 4, "repeat": num_source_domains - 1},
            "high": {"N_oc": 5, "repeat": num_source_domains - 1},
            "full": {"N_oc": num_classes, "repeat": num_source_domains},
        }

        special_class_overlap = {
            "0": [[0, 1], [2, 3], [4, 5, 6]],
            "33": [[0, 1, 2], [2, 3, 4], [4, 5, 6]],
            "66": [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 0]],
            "100": [list(range(7)), list(range(7)), list(range(7))],
        }

        if overlap not in SPECIAL_OVERLAP_TYPES:
            domain_classes = get_domain_classes(
                N_c = num_classes,
                N_oc = overlap_config[overlap]["N_oc"],
                repeat = overlap_config[overlap]["repeat"],
                N_s = num_source_domains,
                seed = overlap_seed
            )
        else:
            domain_classes = special_class_overlap[overlap]

        super().__init__(self.dir, test_envs, hparams['data_augmentation'], 
                         hparams, domain_classes)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams, overlap, overlap_seed):
        self.dir = os.path.join(root, "office_home/")
        num_source_domains = 3
        num_classes = 65
        overlap_config = {
            "none": {"N_oc": 0, "repeat": num_source_domains - 1},
            "low": {"N_oc": 25, "repeat": num_source_domains - 1},
            "mid": {"N_oc": 40, "repeat": num_source_domains - 1},
            "high": {"N_oc": 50, "repeat": num_source_domains - 1},
            "full": {"N_oc": num_classes, "repeat": num_source_domains},
        }

        special_class_overlap = {
            "0": [list(range(0,22)), list(range(22,44)), list(range(44, 65))],
            "33": [list(range(0,30)), list(range(14,44)), list(range(35, 65))], # 25/65
            "66": [list(range(0,38)), list(range(5,44)), list(range(27, 65))], # 50/65
            "100": [list(range(65)), list(range(65)), list(range(65))],
        }

        if overlap not in SPECIAL_OVERLAP_TYPES:
            domain_classes = get_domain_classes(
                N_c = num_classes,
                N_oc = overlap_config[overlap]["N_oc"],
                repeat = overlap_config[overlap]["repeat"],
                N_s = num_source_domains,
                seed = overlap_seed
            )
        else:
            domain_classes = special_class_overlap[overlap]

        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams, 
                         domain_classes)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)

class DomainBedWILDSEnvironment(WILDSEnvironment):
    """
    Custom version of WILDSEnvironment to allow class filtering
    """
    def __init__(
            self, 
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None,
            remove_classes: List[int] = [],
        ):
        super().__init__(wilds_dataset, metadata_name, metadata_value, transform)
        
        # Remove specified classes
        old_indices = self.indices
        self.indices = []

        for index in old_indices:
            if self.dataset.y_array[index] not in remove_classes:
                self.indices.append(index)

        self.targets = self.dataset.y_array[self.indices].numpy()

class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))

class DomainBedWILDSDataset(MultipleDomainDataset):
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams, environments, domain_class_filter: List[List[int]] = None):
        super().__init__()

        num_envs = len(environments) 
        
        assert len(test_envs) == 1, "Not performing leave-one-domain-out validation"

        if isinstance(dataset, Camelyon17Dataset):  
            self.idx_to_class = {0: "No Tumour", 1: "Tumour"}
            self.num_classes = 2
        elif isinstance(dataset, FMoWDataset):
            categories = ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]
            self.idx_to_class = {}
            [self.idx_to_class.update({idx: category}) \
                for idx, category in enumerate(categories)]
            self.num_classes = 62

        if domain_class_filter is None:
            domain_class_filter = [list(range(self.num_classes)) for _ in range(num_envs-1)]

        self.overlapping_classes = self.get_overlapping_classes(domain_class_filter, self.num_classes)

        # Dynamically associate a filter with a domain except for test_envs[0]
        num_filters = len(domain_class_filter)
        assert num_envs-1 == num_filters # b/c exempt first test env
        shift_filter = list(range(num_filters)) + list(range(num_filters))
        shift_filter = shift_filter[test_envs[0]: test_envs[0] + num_filters]

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            # setup class filtering
            if i not in test_envs:
                filter = domain_class_filter[shift_filter.pop()]
                all_classes = set(list(self.idx_to_class.keys())) 
                remove_classes = list(all_classes - set(filter))

                env_dataset = DomainBedWILDSEnvironment(
                    dataset, metadata_name, metadata_value, env_transform,
                    remove_classes)

                env_dataset.is_test_env = False
                env_dataset.allowed_classes = filter
                env_dataset.remove_classes = remove_classes
            else:
                env_dataset = DomainBedWILDSEnvironment(
                    dataset, metadata_name, metadata_value, env_transform)

                env_dataset.is_test_env = True
                env_dataset.allowed_classes = list(range(self.num_classes))
                env_dataset.remove_classes = []

            env_dataset.env_name = environments[i]
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))

    def get_overlapping_classes(self, class_split: List[List[int]], num_classes: int) -> List[int]:
        """ 
        Return the classes in multiple domains.
        """
        overlap = np.zeros(num_classes)
        for data in class_split:
            np.add.at(overlap, data, 1)

        overlapping_classes = list(np.where(overlap>1)[0])

        return overlapping_classes
    

class WILDSCamelyon(DomainBedWILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams, class_overlap_id: int = 100):
        dataset = Camelyon17Dataset(root_dir=root)
        self.class_overlap = {
            0: [[0], [0], [1], [1]],
            33: [[0,1], [0], [1,0], [1]],
            100: [[0,1], [0,1], [0,1], [0,1]],
        }
        self.environment_names = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams,
            self.environment_names, self.class_overlap[class_overlap_id])


class WILDSFMoW(DomainBedWILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams,
            self.ENVIRONMENTS)

