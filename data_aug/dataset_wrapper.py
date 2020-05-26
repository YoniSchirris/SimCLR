import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets
from data_aug.dataset_msi import PreProcessedMSIDataset as dataset_msi


np.random.seed(0)


class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s, data, path_to_msi_data, data_fraction=1):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.path_to_msi_data = path_to_msi_data
        self.input_shape = eval(input_shape)
        self.data = data
        self.data_fraction = data_fraction
        self.train_length = "Not yet initialized"

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()

        if self.data == 'stl10':
            train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
                                       transform=SimCLRDataTransform(data_augment))
        elif self.data == 'msi':
            # train_dataset = custom_histo_dataset
            #TODO using a fraction of the data at this moment
            train_dataset = dataset_msi(root_dir=self.path_to_msi_data, transform=SimCLRDataTransform(data_augment), data_fraction=self.data_fraction)
        else:
            print(f'{self.data} is not an existing data type at this moment. Please check this file to see what you can use')

        self.train_length = train_dataset.__len__()
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader
    
    def get_train_length(self):
        return self.train_length

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.

        # TODO added ToPILImage for it to work with my MSI dataset. See if this breaks the 
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.ToPILImage(),
                                              transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                                              transforms.ToTensor()])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
