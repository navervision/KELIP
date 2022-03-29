"""
KELIP
Copyright (c) 2022-present NAVER Corp.
Apache-2.0
"""
import os
import sys
import json
import torch
import kelip
import gradio as gr

def load_model():
    model, preprocess_img, tokenizer = kelip.build_model('ViT-B/32')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    model_dict = {'model': model,
                  'preprocess_img': preprocess_img,
                  'tokenizer': tokenizer
                  }
    return model_dict

def classify(img, user_text):
    preprocess_img = model_dict['preprocess_img']

    input_img = preprocess_img(img).unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_img = input_img.to(device)

    # extract image features
    with torch.no_grad():
        image_features = model_dict['model'].encode_image(input_img)

        # extract text features
        user_texts = user_text.split(',')
        if user_text == '' or user_text.isspace():
            user_texts = []

        input_texts = model_dict['tokenizer'].encode(user_texts)
        if torch.cuda.is_available():
            input_texts = input_texts.cuda()
        text_features = model_dict['model'].encode_text(input_texts)

    # l2 normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(len(user_texts))
    result = {}
    for value, index in zip(values, indices):
        result[user_texts[index]] = value.item()
    print(result)

    return result

if __name__ == '__main__':
    global model_dict

    model_dict = load_model()
    
    # define gradio demo
    inputs = [gr.inputs.Image(type="pil", label="Image"),
              gr.inputs.Textbox(lines=5, label="Caption"),
              ]

    outputs = ['label']

    title = "KELIP"
    description = "Zero-shot classification with KELIP -- Korean and English bilingual contrastive Language-Image Pre-training model that is trained with collected 1.1 billion image-text pairs (708 million Korean and 476 million English).<br> <br><a href='https://arxiv.org/abs/2203.14463' target='_blank'>Arxiv</a> | <a href='https://github.com/navervision/KELIP' target='_blank'>Github</a>"
    examples = [
    ["demo/images/squid_sundae.jpg", "오징어 순대,김밥,순대,떡볶이"],
    ["demo/images/seokchon_lake.jpg", "평화의문,올림픽공원,롯데월드,석촌호수"],
    ["demo/images/seokchon_lake.jpg", "spring,summer,autumn,winter"],
    ["demo/images/dog.jpg", "a dog,a cat,a tiger,a rabbit"],
    ]

    article = ""

    gr.Interface(classify,
                 inputs,
                 outputs,
                 title=title,
                 description=description,
                 examples=examples,
                 article=article
                 ).launch(server_name="0.0.0.0",server_port=10000)
