import os
import torch
import numpy as np
from torch.utils.data import Dataset
from custom_util.misc import IMG_MEAN, IMG_STD
import sys
OPEN_CLIP_SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'open_clip', 'src')
sys.path.insert(0, OPEN_CLIP_SRC_PATH)
print(f"Adding {OPEN_CLIP_SRC_PATH} to sys.path")
from open_clip import get_tokenizer
import open_clip
print(open_clip.__path__)
import json
import torchvision.transforms as T


def balance_dataset(img_paths, labels):
    
    def get_list_at_idxs(list, idxs):
        return [list[i] for i in idxs]

    label_types = sorted(np.unique(labels))
    labels = np.array(labels)
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
        updated_label_sizes[label] = len(np.where(np.array(updated_labels) == label)[0])
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
        self.mode = 'binary' if self.n_classes == 2 else 'multiclass'

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

class EchoEmbeddingClassification(Dataset):

    def __init__(self, data_path, df, splits, processor=None, data_path_field='path', **kwargs):
        super().__init__()
        self.df_ = df[df[data_path_field].isin(splits[0])].set_index(data_path_field).loc[splits[0]].reset_index()
        self.studies, self.labels = self.df_[data_path_field].to_list(), self.df_['label'].to_list()
        # append data_path as root dir to volume_paths
        self.embedding_dict = torch.load(data_path)
        self.embedding_dict = {study: emb.cpu() for study, emb in self.embedding_dict.items()}
        self.processor = processor
        
        label_words = list(set(self.labels))
        label_words.sort()
        self.n_classes = len(label_words)
        self.label_dict = {label: i for i, label in enumerate(label_words)}
        print('Label dict:', self.label_dict)
        print(f"Number of studies: {len(self.studies)}, Number of labels: {self.n_classes}, Number of embeddings: {len(self.embedding_dict)}")
        
        self.mode = 'binary' if self.n_classes == 2 else 'multiclass'

    def __len__(self):
        return len(self.studies)
                
    def __getitem__(self, item):
        study = self.studies[item]
        label = self.label_dict[self.labels[item]] if not isinstance(self.labels[item], np.ndarray) else self.labels[item]
        embedding = self.embedding_dict[study]
        label = torch.tensor(label).long()
        
        return embedding, label


class EchoViewClassification(Dataset):
    
    def __init__(self, data_path, df, splits, processor=None, data_path_field='path', **kwargs):
        super().__init__()
        self.df_ = df[df[data_path_field].isin(splits[0])].set_index(data_path_field).loc[splits[0]].reset_index()
        self.videos, self.labels = self.df_[data_path_field].to_list(), self.df_['label'].to_list()
        # append data_path as root dir to volume_paths
        self.data_dir = data_path

        label_words = list(set(self.labels))
        label_words.sort()
        self.n_classes = len(label_words)
        self.label_dict = {label: i for i, label in enumerate(label_words)}
        print('Label dict:', self.label_dict)
        print(f"Number of videos: {len(self.videos)}, Number of labels: {self.n_classes}, Number of embeddings: {len(self.data_dir)}")

        self.mode = 'binary' if self.n_classes == 2 else 'multiclass'

    def __len__(self):
        return len(self.videos)
                
    def __getitem__(self, item):
        video = self.videos[item]
        label = self.label_dict[self.labels[item]] if not isinstance(self.labels[item], np.ndarray) else self.labels[item]
        embedding = torch.load(os.path.join(self.data_dir, video))
        label = torch.tensor(label).long()
        
        return embedding, video


class EchoViewClassificationPredict(Dataset):
    def __init__(self, data_path, df, splits, processor=None, data_path_field='path', **kwargs):
        super().__init__()
        self.df_ = df[df[data_path_field].isin(splits[0])].set_index(data_path_field).loc[splits[0]].reset_index()
        self.videos, self.video_ids = self.df_[data_path_field].to_list(), self.df_['video_id'].to_list()
        with open(data_path, 'r') as f:
            self.data = json.load(f) # {"embedding_path": embedding_path, "label_dict" = label_dict, "id_dict": id_dict}
        self.embedding_dir = self.data.get("embedding_dir", '')
        self.label_dict = self.data.get("label_dict", {})
        self.id_to_video = self.data.get("id_to_video", {})

        self.n_classes = len(self.label_dict)
        self.pred_to_label = [label for label in self.label_dict]
        print('Label dict:', self.label_dict)
        print('Pred to Label:', self.pred_to_label)
        print(f"Number of videos: {len(self.videos)}, Number of labels: {self.n_classes}, Number of embeddings: {len(self.embedding_dir)}")
        
        self.mode = 'binary' if self.n_classes == 2 else 'multiclass'
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, item):
        video = self.videos[item]
        embedding = torch.load(os.path.join(self.embedding_dir, video))
        video_id = self.video_ids[item]
        video_id = torch.tensor(video_id).long()
        return embedding, video_id


class EchoZeroShotClassification(Dataset):
    
    def __init__(self, data_path, df, splits, processor=None, data_path_field='path', prompt_path='', task_name="LHF", clip_model_name='', **kwargs):
        super().__init__()
        self.df_ = df[df[data_path_field].isin(splits[0])].set_index(data_path_field).loc[splits[0]].reset_index()
        self.studies, self.labels = self.df_[data_path_field].to_list(), self.df_['label'].to_list()
        # append data_path as root dir to volume_paths
        self.embedding_dict = torch.load(data_path)
        self.embedding_dict = {study: emb.cpu() for study, emb in self.embedding_dict.items()}
        self.processor = processor
        
        prompt_dict = json.load(open(prompt_path, 'r'))
        assert task_name in prompt_dict, f"Task name {task_name} not found in prompt dictionary."
        prompts = prompt_dict[task_name]
        tokenizer = get_tokenizer(clip_model_name)
        self.text_tokens = tokenizer(prompts)
        
        label_words = list(set(self.labels))
        label_words.sort()
        self.n_classes = len(label_words)
        self.label_dict = {label: i for i, label in enumerate(label_words)}
        print('Label dict:', self.label_dict)
        print(f"Number of studies: {len(self.studies)}, Number of labels: {self.n_classes}, Number of embeddings: {len(self.embedding_dict)}")
        
        self.mode = 'binary' if self.n_classes == 2 else 'multiclass'

    def __len__(self):
        return len(self.studies)
                
    def __getitem__(self, item):
        study = self.studies[item]
        label = self.label_dict[self.labels[item]] if not isinstance(self.labels[item], np.ndarray) else self.labels[item]
        embedding = self.embedding_dict[study]
        label = torch.tensor(label).long()
        
        return (embedding, self.text_tokens), label


class EchoBiomedGPTClassification(Dataset):
    
    def __init__(self, data_path, df, splits, processor=None, data_path_field='path', prompt_path='', task_name="LHF", **kwargs):
        super().__init__()
        self.df_ = df[df[data_path_field].isin(splits[0])].set_index(data_path_field).loc[splits[0]].reset_index()
        self.studies, self.labels = self.df_[data_path_field].to_list(), self.df_['label'].to_list()
        # append data_path as root dir to volume_paths
        self.data_dir = data_path
        
        from transformers import OFATokenizer
        self.tokenizer = OFATokenizer.from_pretrained("./BiomedGPT-Base-Pretrained")
        prompt_dict = json.load(open(prompt_path, 'r'))
        assert task_name in prompt_dict, f"Task name {task_name} not found in prompt dictionary."
        prompts = prompt_dict[task_name]
        print(f"Using prompt: {prompts}")
        self.input_ids = self.tokenizer([prompts], return_tensors="pt").input_ids
        self.decoder_input_ids = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long)

        label_words = list(set(self.labels))
        label_words.sort()
        self.n_classes = len(label_words)
        self.label_dict = {label: i for i, label in enumerate(label_words)}
        print('Label dict:', self.label_dict)
        print(f"Number of studies: {len(self.studies)}, Number of labels: {self.n_classes}")

        self.mode = 'binary' if self.n_classes == 2 else 'multiclass'
        
        self.mean, self.std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((480, 480)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

    def __len__(self):
        return len(self.studies)
                
    def __getitem__(self, item):
        study = self.studies[item]
        label = self.label_dict[self.labels[item]] if not isinstance(self.labels[item], np.ndarray) else self.labels[item]
        video_list = os.listdir(os.path.join(self.data_dir, study))
        x = []
        for i in range(2):
            video = np.random.choice(video_list)
            img = np.load(os.path.join(self.data_dir, study, video))
            # print(f"Loaded image shape: {img.shape}")
            index = np.random.choice(img.shape[1], size=1)[0]
            img = img[:, index, :, :] # (C, H, W)
            img = np.transpose(img, (1, 2, 0)) # (H, W, C)
            # print(f"Selected frame shape: {img.shape}")
            
            # resize to 480 x 480 and normalize
            img = self.transform(img) # (1, 3, 480, 480)
            # print(f"Transformed image shape: {img.shape}")
            x.append(img)
        
        
        label = torch.tensor(label).long()
        
        return (x[0], x[1], self.input_ids, self.decoder_input_ids), label

def collate_fn_biomedgpt(batch):
    xs, labels = zip(*batch)
    x1 = torch.stack([x[0] for x in xs], dim=0) # (B, 3, 480, 480)
    x2 = torch.stack([x[1] for x in xs], dim=0) # (B, 3, 480, 480)
    input_ids = torch.cat([x[2] for x in xs], dim=0) # (B, seq_len) they should have the same seq_len
    decoder_input_ids = torch.cat([x[3] for x in xs], dim=0) # (B, 1)
    labels = torch.stack(labels, dim=0)
    return (x1, x2, input_ids, decoder_input_ids), labels

def get_collate_fn(dataclass):
    if dataclass == "EchoBiomedGPTClassification":
        return collate_fn_biomedgpt