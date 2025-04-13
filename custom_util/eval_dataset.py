import os
import torch
import numpy as np
from torch.utils.data import Dataset
from custom_util.misc import IMG_MEAN, IMG_STD


def balance_dataset(img_paths, labels):
    
    def get_list_at_idxs(list, idxs):
        return [list[i] for i in idxs]

    label_types = sorted(np.unique(labels))
    label_sizes = {label: len(np.where(labels == label)[0]) for label in label_types}
    print(f"Starting with: {'  '.join([f'{label} (N={label_sizes[label]})' for label in label_types])}")
    tgt_size = max(label_sizes.values())
    
    updated_img_paths = []
    updated_labels = []
    updated_label_sizes = {}
    for label in label_types:
        data_pool = np.where(labels == label)[0]  # idxs to work with
        if len(data_pool) == tgt_size:
            updated_img_paths.extend(get_list_at_idxs(img_paths, data_pool))
            updated_labels.extend(get_list_at_idxs(labels, data_pool))
        elif len(data_pool) < tgt_size:
            updated_img_paths.extend(get_list_at_idxs(img_paths, data_pool))
            updated_labels.extend(get_list_at_idxs(labels, data_pool))
            rdm_sample = np.random.choice(data_pool, size=tgt_size-len(data_pool))
            updated_img_paths.extend(get_list_at_idxs(img_paths, rdm_sample))
            updated_labels.extend(get_list_at_idxs(labels, rdm_sample))
        else:
            raise ValueError(f"Unexpected size {len(data_pool)} for label {label}, with max size {tgt_size} for label {label_types[0]}.")
        updated_label_sizes[label] = len(np.where(updated_labels == label)[0])
    print(f"\tUpdated to: {'  '.join([f'{label} (N={updated_label_sizes[label]})' for label in label_types])}")
    return updated_img_paths, updated_labels
    

class ClassificationDataset3D(Dataset):
    def __init__(self,
                 data_path, 
                 df, 
                 splits, 
                 processor=None, 
                 label_processor=None,
                 normalize_vol=True, 
                 make_three_channel=False,
                 data_path_field='processed_ct_volume_path'):

        #self.df_ = df[df[data_path_field].isin(splits[0])].set_index(data_path_field).reindex(splits[0]).reset_index()
        # Since splits[0] contains repeated values, using .reindex(splits[0]) can cause issues if the index is not unique. To avoid potential errors, we should ensure that the reindexing process correctly handles duplicates.
        self.df_ = df[df[data_path_field].isin(splits[0])].set_index(data_path_field).loc[splits[0]].reset_index()
        self.volume_paths, self.labels = self.df_[data_path_field].to_list(), self.df_['label'].to_list()
        # append data_path as root dir to volume_paths
        self.volume_paths = [os.path.join(data_path, path) for path in self.volume_paths]
        self.processor = processor
        self.normalize_vol = normalize_vol
        self.make_3ch = make_three_channel

        if label_processor is not None:
            label_processor(self.labels)
        else:
            label_words = list(set(self.labels))
            label_words.sort()
            self.n_classes = len(label_words)
            self.label_dict = {label: i for i, label in enumerate(label_words)}
        print('Label dict:', self.label_dict)

    def norm_vol_orientation(self, vol_tensor):
        vol_tensor = vol_tensor.permute(0, 2, 1, 3).flip(2)
        return vol_tensor
    
    def img_to_tensor(self, img):
        return torch.tensor(img).unsqueeze(0).float()

    def label_to_tensor(self, label):
        return torch.tensor(label).long()

    def __len__(self):
        return len(self.volume_paths)
                
    def __getitem__(self, item):
        #try:
        #    img = np.load(self.volume_paths[item])
        #except:
        #    print(self.volume_paths[item])
        #    exit(0)
        
        img = np.load(self.volume_paths[item])
        label = self.label_dict[self.labels[item]] if not isinstance(self.labels[item], np.ndarray) else self.labels[item]
        vol_tensor = self.img_to_tensor(img)
        if self.make_3ch:
            vol_tensor = vol_tensor.repeat(3, 1, 1, 1) # shape: C x T x W x H 
        # orient to align with in house data
        vol_tensor = self.norm_vol_orientation(vol_tensor)
        # normalize the input volume
        if self.normalize_vol:
            if torch.max(vol_tensor) > 1:
                vol_tensor = vol_tensor / 255.0
            vol_tensor = (vol_tensor - IMG_MEAN) / IMG_STD
        if self.processor is not None:
            vol_tensor = self.processor({"pixel_values": vol_tensor})["pixel_values"]
        label = self.label_to_tensor(label)

        # save vol_tensor into gif for visual inspect
        # from PIL import Image
        # # vol_tensor = 255 * (vol_tensor * IMG_STD + IMG_MEAN)
        # vol_tensor = 122.5 * (vol_tensor + 1)
        # images = []
        # for i in range(vol_tensor.shape[1]):
        #     images.append(Image.fromarray((vol_tensor[0, i, :, :].numpy())))
        # images[0].save('vol_tensor.gif', save_all=True, append_images=images[1:], duration=200, loop=0)
        # print('vol_tensor:', vol_tensor.shape, 'label:', label)
        # print(self.volume_paths[item])
        # raise ValueError('stop here')
        
        return vol_tensor, label


class EchoData(ClassificationDataset3D):
    
    def __init__(self, data_path, df, splits, processor=None, normalize_vol=True, make_three_channel=False, data_path_field='processed_ct_volume_path', **kwargs):
        super().__init__(data_path, df, splits, processor, None, normalize_vol, make_three_channel, data_path_field)
        self.mode = 'binary'

    def img_to_tensor(self, img):
        """
        out: C x T x W x H
        """
        if len(img.shape) == 3:
            return torch.tensor(img).unsqueeze(0).float() #.permute(0, 3, 1, 2)
        #print(torch.tensor(img).float().permute(0, 3, 1, 2).shape)
        #exit(0)
        return torch.tensor(img).float() #.permute(0, 3, 1, 2)
    
    def norm_vol_orientation(self, vol_tensor):
        return vol_tensor