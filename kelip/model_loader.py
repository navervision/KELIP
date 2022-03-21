"""
KELIP
Copyright (c) 2022-present NAVER Corp.
Apache-2.0
"""
import torch
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from .model import CLIP

class Tokenizer(object):
    def __init__(self, revision):
        self.tokenizer = AutoTokenizer.from_pretrained('navervision/KELIP', revision=revision)
        self.pad_token = self.tokenizer.convert_tokens_to_ids('[PAD]')
    
    def __len__(self):
        return len(self.tokenizer)

    def encode(self, texts, context_length=77):
        if isinstance(texts, str):
            texts = [texts.lower()]
        else:
            texts = [text.lower() for text in texts]
        sot_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        eot_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        all_tokens = [[sot_token] + self.tokenizer.encode(text, max_length=10000, truncation=True)[:context_length-2] + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long) + self.pad_token
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

def _convert_to_rgb(image):
    return image.convert('RGB')

class KelipConfig(PretrainedConfig):
    def __init__(
            self,
            embed_dim=0,
            # vision
            image_resolution = 0,
            vision_layers = [1,2,3],
            vision_width=0,
            vision_patch_size=0,
            # text
            context_length=0,
            vocab_size=0,
            transformer_width=0,
            transformer_heads=0,
            transformer_layers=0,
            **kwargs,
            ):
        self.embed_dim = embed_dim
        # vision
        self.image_resolution = image_resolution
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        # text
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        super().__init__(**kwargs)


class KelipModel(PreTrainedModel):
    config_class = KelipConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = CLIP(
                config.embed_dim,
                config.image_resolution,
                config.vision_layers,
                config.vision_width,
                config.vision_patch_size,
                config.context_length,
                config.vocab_size,
                config.transformer_width,
                config.transformer_heads,
                config.transformer_layers)

    def encode_image(self, image, l2norm=False):
        image_features = self.model.encode_image(image)
        if l2norm:
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def encode_text(self, text, l2norm=False):
        text_features = self.model.encode_text(text)
        if l2norm:
            text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    def forward(self, image, text):
        image_features = self.encode_image(image, l2norm=True)
        text_features = self.encode_text(text, l2norm=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

def build_model(model_name='ViT-B/32'):
    revision = model_name.replace('/', '-').lower()
    tokenizer = Tokenizer(revision)
    model = KelipModel.from_pretrained('navervision/KELIP', revision=revision)
    preprocess = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        _convert_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    return model, preprocess, tokenizer


if __name__ == '__main__':
    from PIL import Image
    from urllib.request import urlretrieve

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess_img, tokenizer = build_model('ViT-B/32')
    model = model.to(device)
    model.eval()

    #urlretrieve('https://upload.wikimedia.org/wikipedia/commons/1/1d/Lotte_World_Groupe_F_Seoul.jpg', 'test.jpg')
    image = preprocess_img(Image.open('test.jpg')).unsqueeze(0).to(device)
    text = tokenizer.encode(['불꽃놀이', '야경', '롯데타워', '석촌호수']).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image, l2norm=True)
        text_features = model.encode_text(text, l2norm=True)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1)

    print("Label probs:", probs)
