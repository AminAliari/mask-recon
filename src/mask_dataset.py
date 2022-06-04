import os
import random
from glob import glob

import PIL
import torch
import numpy as np
from torchvision import transforms


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s + pad)
                             for s, pad in zip(image.size, [p_left, p_top])]
        padding = [p_left, p_top, p_right, p_bottom]
        return transforms.functional.pad(image, padding, 0, 'constant')


class RotateRight:
    def __call__(self, x):
        return transforms.functional.rotate(x, 30)


class RotateLeft:
    def __call__(self, x):
        return transforms.functional.rotate(x, -30)


class MaskDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, img_dim):
        self.root_dir = root_dir
        self.img_dim = img_dim

        self.labels = {'no-mask': 0, 'surgical': 1,
                       'cloth': 2, 'n95': 3, 'n95-valve': 4}
        self.label_to_str = ['no-mask', 'surgical',
                             'cloth', 'n95', 'n95-valve']

        self.gender_types = ['men', 'women']
        self.age_types = ['kid', 'adult', 'old']

        self.transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(self.img_dim),
            transforms.ToTensor(),
        ])
        self.flip_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(self.img_dim),
            transforms.RandomHorizontalFlip(1),
            transforms.ToTensor(),
        ])
        self.center_crop_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(self.img_dim),
            transforms.CenterCrop(196),
            transforms.Resize(self.img_dim),
            transforms.ToTensor(),
        ])
        self.rotate_45_left_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(self.img_dim),
            RotateLeft(),
            transforms.ToTensor(),
        ])
        self.rotate_45_right_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(self.img_dim),
            RotateRight(),
            transforms.ToTensor(),
        ])
        self.random_perspective_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(self.img_dim),
            transforms.RandomPerspective(0.2, 1),
            transforms.ToTensor(),
        ])

        # find images path and their classes in all subdirs.
        self.data = []
        self.n95_data = []
        self.cloth_data = []
        self.no_mask_data = []
        self.surgical_data = []
        self.n95_valve_data = []

        self.n95_data_augmented = []
        self.n95_data_augmented_men = []
        self.cloth_data_augmented = []
        self.surgical_data_augmented = []
        self.n95_valve_data_augmented = []
        self.metadata = np.loadtxt(os.path.join(
            root_dir, 'metadata.csv'), dtype=int, skiprows=1, delimiter=',')[:, 1:]

        for path1 in glob(f'{self.root_dir}/*/'):
            for path2 in glob(f'{path1}/*/'):
                label = os.path.basename(os.path.normpath(path2))
                for img_path in glob(f'{path2}/*'):
                    id = int(os.path.basename(
                        img_path).split('_')[1].split('.')[0])
                    gender, age = self.metadata[id]
                    if 'no-mask' in label:
                        self.no_mask_data.append(
                            [img_path, label, self.transform, gender, age])
                    if 'cloth' in label:
                        self.cloth_data.append(
                            [img_path, label, self.transform, gender, age])
                        self.cloth_data_augmented.append(
                            [img_path, label, self.flip_transform, gender, age])
                    if 'surgical' in label:
                        self.surgical_data.append(
                            [img_path, label, self.transform, gender, age])
                        self.surgical_data_augmented.append(
                            [img_path, label, self.flip_transform, gender, age])
                    if 'n95-valve' in label:
                        self.n95_valve_data.append(
                            [img_path, label, self.transform, gender, age])
                        self.n95_valve_data_augmented.append(
                            [img_path, label, self.flip_transform, gender, age])
                        self.n95_valve_data_augmented.append(
                            [img_path, label, self.center_crop_transform, gender, age])
                        self.n95_valve_data_augmented.append(
                            [img_path, label, self.rotate_45_left_transform, gender, age])
                        self.n95_valve_data_augmented.append(
                            [img_path, label, self.rotate_45_right_transform, gender, age])
                        self.n95_valve_data_augmented.append(
                            [img_path, label, self.random_perspective_transform, gender, age])
                    elif 'n95' in label:
                        self.n95_data.append(
                            [img_path, label, self.transform, gender, age])
                        self.n95_data_augmented.append(
                            [img_path, label, self.flip_transform, gender, age])
                        self.n95_data_augmented.append(
                            [img_path, label, self.center_crop_transform, gender, age])
                        if gender == 'men':
                            self.n95_data_augmented_men.append(
                                [img_path, label, self.rotate_45_left_transform, gender, age])
                            self.n95_data_augmented_men.append(
                                [img_path, label, self.rotate_45_right_transform, gender, age])
                            self.n95_data_augmented_men.append(
                                [img_path, label, self.random_perspective_transform, gender, age])

        random.shuffle(self.n95_data_augmented)
        random.shuffle(self.cloth_data_augmented)
        random.shuffle(self.n95_valve_data_augmented)
        random.shuffle(self.surgical_data_augmented)

        self.no_mask_data = self.no_mask_data[:600]
        self.n95_data.extend(
            self.n95_data_augmented_men[:600 - len(self.n95_data)])
        self.n95_data.extend(
            self.n95_data_augmented[:600 - len(self.n95_data)])
        self.n95_valve_data.extend(
            self.n95_valve_data_augmented[:600 - len(self.n95_valve_data)])
        self.surgical_data.extend(
            self.surgical_data_augmented[:600 - len(self.surgical_data)])
        self.cloth_data.extend(
            self.cloth_data_augmented[:600 - len(self.cloth_data)])

        self.data.extend(self.cloth_data)
        self.data.extend(self.no_mask_data)
        self.data.extend(self.n95_data)
        self.data.extend(self.n95_valve_data)
        self.data.extend(self.surgical_data)

    def __getitem__(self, idx):
        img_path, label, transform, gender, age = self.data[idx]
        img = PIL.Image.open(img_path).convert('RGB')
        img_tensor = transform(img)

        label_tensor = torch.tensor(self.labels[label])

        return img_tensor, label_tensor, gender, age

    def __len__(self):
        return len(self.data)

    def __get_train_valid_indices(self, split_ratio):
        data_size = self.__len__()
        train_size = int(data_size * split_ratio)
        valid_size = data_size - train_size

        per_label_train_size = train_size // 5
        per_label_valid_size = valid_size // 5

        train_indices = []
        valid_indices = []

        for i in range(0, 5):
            r = i * 600
            train_indices.extend(range(r, r + per_label_train_size))
            valid_indices.extend(
                range(r + per_label_train_size, r + per_label_train_size + per_label_valid_size))

        return train_indices, valid_indices

    def get_data_loaders(self, batch_size, use_shuffle, split_ratio):
        train_indices, valid_indices = self.__get_train_valid_indices(
            split_ratio=split_ratio)

        train_dataset = torch.utils.data.Subset(self, train_indices)
        valid_dataset = torch.utils.data.Subset(self, valid_indices)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=use_shuffle)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader
