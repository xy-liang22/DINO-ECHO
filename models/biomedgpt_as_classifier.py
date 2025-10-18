import torch
from torch import nn
from transformers import OFATokenizer, OFAModel

class BiomedGPTClassifier(nn.Module):
    def __init__(self,
                 biomedgpt_path: str="./BiomedGPT-Base-Pretrained", num_classes: int=2,
                 **kwargs):
        
        super(BiomedGPTClassifier, self).__init__()
        assert num_classes == 2, "Currently only binary classification is supported."
        self.biomedgpt_model = OFAModel.from_pretrained(f"./{biomedgpt_path}")
        self.tokenizer = OFATokenizer.from_pretrained(f"./{biomedgpt_path}")
        self.yes_token_id = self.tokenizer.encode('yes', add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode('no', add_special_tokens=False)[0]
        print(f"BiomedGPTClassifier initialized with BiomedGPT model '{biomedgpt_path}'. Yes token id: {self.yes_token_id}, No token id: {self.no_token_id}")

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()
    
    def get_num_layers(self):
        return 1


    def forward(self, x):
        assert len(x) ==4, "Input should be a tuple of (images, images_2, input_ids, decoder_input_ids)"
        images = x[0]
        # images_2 = x[1]
        # texts = x[2]
        
        # input_ids = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(images.device)
        input_ids = x[2]
        decoder_input_ids = x[3]
        
        batch_size = input_ids.shape[0]
        num_patches = (images.shape[2] // 16) ** 2  # Assuming patch size of 16
        # patch_masks = torch.ones((batch_size, num_patches), dtype=torch.bool, device=images.device)  # All patches are valid
        
        # decoder_input_ids = torch.tensor([[self.tokenizer.bos_token_id]] * batch_size, dtype=torch.long).to(images.device)
        # print(f"Input IDs shape: {input_ids.shape}, Decoder Input IDs shape: {decoder_input_ids.shape}")
        # print(f"Images shape: {images.shape}, Images_2 shape: {images_2.shape}")
        # print(f"Patch masks shape: {patch_masks.shape}")

        # assert torch.all(input_ids < self.tokenizer.vocab_size), "Input IDs contain values outside the vocabulary range."
        # assert torch.all(decoder_input_ids < self.tokenizer.vocab_size), "Decoder Input IDs contain values outside the vocabulary range."
        # assert self.yes_token_id < self.tokenizer.vocab_size, "Yes token ID is outside the vocabulary range."
        # assert self.no_token_id < self.tokenizer.vocab_size, "No token ID is outside the vocabulary range."
        
        # print(f"Min and Max of input_ids: {input_ids.min().item()}, {input_ids.max().item()}")
        # print(f"Min and Max of decoder_input_ids: {decoder_input_ids.min().item()}, {decoder_input_ids.max().item()}")
        # print(f"Vocabulary size: {self.tokenizer.vocab_size}")
        # print(f"Min and Max of images: {images.min().item()}, {images.max().item()}")
        # print(f"Min and Max of images_2: {images_2.min().item()}, {images_2.max().item()}")
        
        outputs = self.biomedgpt_model(
            input_ids=input_ids,
            patch_images=images,
            # patch_images_2=images_2,
            # patch_masks=patch_masks,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )
        # print(f"Model outputs obtained.")
        logits = outputs.logits  # (B, seq_len, vocab_size)
        logits = logits[:, -1, :]  # (B, vocab_size)
        probs = torch.softmax(logits, dim=-1)  # (B, vocab_size)
        combined_no_yes_prob = probs[:, [self.no_token_id, self.yes_token_id]]  # (B, 2)
        return combined_no_yes_prob
    

def biomedgpt_classifier(**kwargs):
    model = BiomedGPTClassifier(biomedgpt_path="./BiomedGPT-Base-Pretrained",**kwargs)
    return model