import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
import csv
import numpy as np
from torch.utils.data.sampler import Sampler


def pm_folders(test_person):
    data_folder = ''

    metatrain_character_folders = {}
    metaval_character_folders = {}
    for person in [f for f in os.listdir(data_folder) if f != '.DS_Store']:
        person_folder = os.path.join(data_folder, person)
        if person == test_person:
            metaval_character_folders[person] = {}
        else:
            metatrain_character_folders[person] = {}
        for activity in [p for p in os.listdir(person_folder) if p != '.DS_Store']:
            activity_folder = os.path.join(person_folder, activity)
            if person == test_person:
                metaval_character_folders[person][activity] = []
            else:
                metatrain_character_folders[person][activity] = []
            for item in [a for a in os.listdir(activity_folder) if a != '.DS_Store']:
                if person == test_person:
                    metaval_character_folders[person][activity].append(os.path.join(activity_folder, item))
                else:
                    metatrain_character_folders[person][activity].append(os.path.join(activity_folder, item))
    return metatrain_character_folders, metaval_character_folders


class PMTask(object):
    def __init__(self, character_folders, num_classes, train_num):
        data_folder = ''
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        # self.test_num = test_num

        labels = dict()
        self.per_class_num = 10000
        for index in range(self.num_classes):
            random_person = random.sample(self.character_folders.keys(), 1)[0]
            random_person_path = os.path.join(data_folder, random_person)
            # class_folders = random.sample(self.character_folders[random_person].keys(), 1)
            # print(class_folders)
            class_folder = os.path.join(random_person_path, str(index))
            labels[class_folder] = index

            self.train_roots = []
            self.test_roots = []

            temp = [os.path.join(class_folder, x) for x in os.listdir(class_folder)]
            if self.per_class_num > len(temp) - train_num:
                self.per_class_num = len(temp) - train_num

        samples = dict()
        for key, item in labels.items():
            temp = [os.path.join(key, x) for x in os.listdir(key)]
            samples[item] = random.sample(temp, len(temp))

            self.train_roots += samples[item][:train_num]
            self.test_roots += samples[item][train_num:train_num+self.per_class_num]

        self.test_num = self.per_class_num

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        return '/'+os.path.join(*sample.split('/')[:-1])


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform  # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class PM(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(PM, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        _data = csv.reader(open(image_root, "r"), delimiter=",")
        for row in _data:
            image = [float(f) for f in row]
            image = np.array(image)
            image = np.reshape(image, (5, 16, 16))
            image = np.swapaxes(image, 1, 2)
            image = np.reshape(image, (1, 80, 16))
            if self.transform is not None:
                image = self.transform(image)
            label = self.labels[idx]
            if self.target_transform is not None:
                label = self.target_transform(label)
            return image, label


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train', shuffle=True):

    dataset = PM(task, split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    return loader
