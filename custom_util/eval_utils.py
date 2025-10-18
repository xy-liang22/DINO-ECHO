import os
import warnings
import numpy as np
import pandas as pd

import monai.transforms as monai_transforms
from sklearn.model_selection import train_test_split


def create_transforms(image_size=256, num_frames=512, max_frames=512, RandFlipd_prob=0.5, RandRotate90d_prob=0.5, dataclass=None, **kwargs):
    train_transform, val_transform = None, None
    if dataclass == "EchoData":
        # create the transform function
        train_transform = monai_transforms.Compose(
            [   
                monai_transforms.ResizeWithPadOrCropd(
                    keys=["pixel_values"], spatial_size=(max_frames, -1, -1)
                ),
                monai_transforms.Resized(
                    keys=["pixel_values"], spatial_size=(num_frames, image_size, image_size), mode=("bilinear")
                ),
            ]
        )
        
        val_transform = monai_transforms.Compose(
            [
                monai_transforms.ResizeWithPadOrCropd(
                    keys=["pixel_values"], spatial_size=(max_frames, -1, -1)
                ),
                monai_transforms.Resized(
                    keys=["pixel_values"], spatial_size=(num_frames, image_size, image_size), mode=("bilinear")
                ),
            ]
        )

    return train_transform, val_transform


def train_splits_bootstrap(train_splits, fold=0):
    np.random.seed(fold)
    # if choose_mini:
    #     mini_indices = np.random.choice(len(train_splits[0]), size=int(len(train_splits[0]) * 0.2), replace=False)
    #     train_splits = ([train_splits[0][i] for i in mini_indices], [train_splits[1][i] for i in mini_indices])
    #     return train_splits
    bootstrap_indices = np.random.choice(len(train_splits[0]), size=len(train_splits[0]), replace=True)
    train_splits = ([train_splits[0][i] for i in bootstrap_indices], [train_splits[1][i] for i in bootstrap_indices])
    return train_splits


def get_random_splits(df, val_r=0.1, test_r=0.1, fold=0, split_dir='', fetch_splits=True, prop=1, pat_strat=True, args=None):
    os.makedirs(split_dir, exist_ok=True)
    # get the split names
    files = os.listdir(split_dir)
    train_name, val_name, test_name = f'train_{fold}.csv', f'val_{fold}.csv', f'test_{fold}.csv'
    # make sure the dataset exists, otherwise create new datasets
    if train_name not in files or val_name not in files or test_name not in files or not fetch_splits:
        if pat_strat:
            patients = df.drop_duplicates('patients')['patients'].to_list()
            train_pat, temp_pat = train_test_split(patients, test_size=(val_r + test_r), random_state=fold)
            val_pat, test_pat = train_test_split(temp_pat, test_size=(test_r / (val_r + test_r)), random_state=fold)
            train_data = df[df['patients'].isin(train_pat)]
            val_data = df[df['patients'].isin(val_pat)]
            test_data = df[df['patients'].isin(test_pat)]
        else:
            train_data, temp_data = train_test_split(df, test_size=(val_r + test_r), random_state=fold)
            val_data, test_data = train_test_split(temp_data, test_size=(test_r / (val_r + test_r)), random_state=fold)
        # sample the training data
        if prop > 0:
            train_data = train_data.sample(frac=prop, random_state=fold).reset_index(drop=True)
        # save datasets
        train_data.to_csv(os.path.join(split_dir, train_name))
        val_data.to_csv(os.path.join(split_dir, val_name))
        test_data.to_csv(os.path.join(split_dir, test_name))
    # load the dataframe
    if pat_strat:
        train_splits = pd.read_csv(os.path.join(split_dir, train_name))['volumes'].to_list()
        val_splits = pd.read_csv(os.path.join(split_dir, val_name))['volumes'].to_list()
        test_splits = pd.read_csv(os.path.join(split_dir, test_name))['volumes'].to_list()
    else:
        train_splits = pd.read_csv(os.path.join(split_dir, train_name))['volumes'].to_list()
        val_splits = pd.read_csv(os.path.join(split_dir, val_name))['volumes'].to_list()
        test_splits = pd.read_csv(os.path.join(split_dir, test_name))['volumes'].to_list()
    return train_splits, val_splits, test_splits


def get_train_bootstrap_splits(df, args=None, fold=0, choose_mini=False, **kwargs):  # test splits are already defined!

    train_md = df[df.split == "train"]
    val_md = df[df.split == "val"]
    test_md = df[df.split == "test"]
    
    train_splits = (train_md['processed_ct_volume_path'].to_list(), train_md['label'].to_list())
    val_splits = (val_md['processed_ct_volume_path'].to_list(), val_md['label'].to_list())
    test_splits = (test_md['processed_ct_volume_path'].to_list(), test_md['label'].to_list())

    # do bootstrap sampling for training
    train_splits = train_splits_bootstrap(train_splits, fold=fold, choose_mini=choose_mini)

    return train_splits, val_splits, test_splits


####### for echo
def get_train_bootstrap_splits_echo(df, args=None, fold=0, **kwargs):  # test splits are already defined!

    train_md = df[df.split == "train"]
    val_md = df[df.split == "val"]
    test_md = df[df.split == "test"]
    
    if args.task_name in ['TV_stenosis', 'PV_stenosis']:
        # sample the ratio of 0 and 1 to be 1:1 in train val and test set
        train_0_md = train_md[train_md.label == 0]
        train_1_md = train_md[train_md.label == 1]
        print(f"Before sampling, train set has {len(train_0_md)} 0 and {len(train_1_md)} 1")
        val_0_md = val_md[val_md.label == 0]
        val_1_md = val_md[val_md.label == 1]
        test_0_md = test_md[test_md.label == 0]
        test_1_md = test_md[test_md.label == 1]
        
        train_1_md = train_1_md.sample(n=len(train_0_md), random_state=fold, replace=True)
        train_md = pd.concat([train_0_md, train_1_md]).reset_index(drop=True)
        print(f"After sampling, train set has {len(train_md[train_md.label == 0])} 0 and {len(train_md[train_md.label == 1])} 1")
        
        val_1_md = val_1_md.sample(n=len(val_0_md), random_state=fold, replace=True)
        val_md = pd.concat([val_0_md, val_1_md]).reset_index(drop=True)
        
        test_1_md = test_1_md.sample(n=len(test_0_md), random_state=fold, replace=True)
        test_md = pd.concat([test_0_md, test_1_md]).reset_index(drop=True)
        
        train_splits = (train_md['path'].to_list(), train_md['label'].to_list())
        val_splits = (val_md['path'].to_list(), val_md['label'].to_list())
        test_splits = (test_md['path'].to_list(), test_md['label'].to_list())
    
    if args.choose_mini:
        val_md = train_md.sample(frac=0.05, random_state=fold)
        train_md = train_md.drop(val_md.index).reset_index(drop=True)
        val_md = val_md.reset_index(drop=True)
        train_md = train_md.sample(frac=0.1, random_state=fold).reset_index(drop=True)
        
    
    if 'label' in train_md.columns:
        train_splits = (train_md['path'].to_list(), train_md['label'].to_list())
        val_splits = (val_md['path'].to_list(), val_md['label'].to_list())
        test_splits = (test_md['path'].to_list(), test_md['label'].to_list())
    else:
        train_splits = (train_md['path'].to_list(), [None] * len(train_md))
        val_splits = (val_md['path'].to_list(), [None] * len(val_md))
        test_splits = (test_md['path'].to_list(), [None] * len(test_md))

    # do bootstrap sampling for training
    train_splits = train_splits_bootstrap(train_splits, fold=fold)

    return train_splits, val_splits, test_splits

def get_echo_splits(df, args=None, fold=0, choose_mini=False, **kwargs):  # test splits are already defined!
    return get_train_bootstrap_splits_echo(df, args=args, fold=fold, choose_mini=choose_mini, **kwargs)

####### end for echo


def get_split_func(dataclass):
    if 'Echo' in dataclass:
        return get_echo_splits
    else:
        return get_random_splits