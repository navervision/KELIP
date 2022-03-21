# KELIP

## USAGE
We provide an easy-to-use KELIP API 
```
$ pip install git+https://github.com/navervision/KELIP.git
```

```python
import kelip
from PIL import Image
from urllib.request import urlretrieve

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess_img, tokenizer = kelip.build_model('ViT-B/32')
model = model.to(device)
model.eval()

urlretrieve('https://upload.wikimedia.org/wikipedia/commons/1/1d/Lotte_World_Groupe_F_Seoul.jpg', 'test.jpg')
image = preprocess_img(Image.open('test.jpg')).unsqueeze(0).to(device)
text = tokenizer.encode(['불꽃놀이', '야경', '롯데타워', '석촌호수']).to(device)
with torch.no_grad():
	image_features = model.encode_image(image, l2norm=True)
	text_features = model.encode_text(text, l2norm=True)

	logits_per_image, logits_per_text = model(image, text)
	probs = logits_per_image.softmax(dim=-1)

print("Label probs:", probs)
```

## Demo
<img src=".github/demo.PNG" height="300">

```
$ pip install gradio
$ python demo/demo_zeroshot.py
```

## License
```
KELIP
Copyright 2022-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
