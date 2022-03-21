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
        result[user_texts[index]] = f"{value.item()*100:.2f}%"

    return result

if __name__ == '__main__':
    print('\tLoading models')

    global model_dict

    model_dict = load_model()
    
    # define gradio demo
    inputs = [gr.inputs.Image(type="pil", label="Image"),
              gr.inputs.Textbox(lines=5, label="Caption"),
              ]

    outputs = gr.outputs.KeyValues()

    title = "Zeroshot classification demo"

    if torch.cuda.is_available():
        demo_status = "Demo is running on GPU"
    else:
        demo_status = "Demo is running on CPU"
    description = f"Details: paper_url. {demo_status}"
    examples = [
    ["demo/images/jkh.png", "장기하,아이유,조인성,마동석"],
    ["demo/images/jkh.png", "눈감았음,눈떴음"],
    ["demo/images/squid_sundae.jpg", "오징어 순대,김밥,순대,떡볶이"],
    ["demo/images/poysian.jpg", "립스틱,분필,야돔"],
    ["demo/images/world_peace_gate.jpg", "평화의문,올림픽공원,롯데월드,석촌호수"],
    ["demo/images/seokchon_lake.jpg", "평화의문,올림픽공원,롯데월드,석촌호수"],
    ["demo/images/hwangchil_tree.jpg", "황칠 나무 묘목,황칠 나무,난,소나무 묘목,야자수"],
    ["demo/images/areca_palm.jpg", "아레카야자,난초,난,식물,장미,야자수,황칠나무"],
    ["demo/images/world_peace_gate.jpg", "봄,여름,가을,겨울"],
    ["demo/images/seokchon_lake.jpg", "봄,여름,가을,겨울"],
    ["demo/images/spring.jpg", "봄,여름,가을,겨울"],
    ["demo/images/summer1.jpg", "봄,여름,가을,겨울"],
    ["demo/images/summer2.jpeg", "봄,여름,가을,겨울"],
    ["demo/images/autumn1.JPG", "봄,여름,가을,겨울"],
    ["demo/images/autumn2.jpg", "봄,여름,가을,겨울"],
    ["demo/images/winter1.jpg", "봄,여름,가을,겨울"],
    ["demo/images/winter2.jpg", "봄,여름,가을,겨울"],
    ["demo/images/airplane.png", "a photo of a airplane.,a photo of a bear.,a photo of a bird.,a photo of a giraffe.,a photo of a car."],
    ["demo/images/airplane.png", "비행기 사진.,곰 사진.,새 사진.,기린 사진.,자동차 사진."],
    ["demo/images/volleyball.png", "a photo of a person volleyball spiking.,a photo of a person jump rope.,a photo of a person soccer penalty.,a photo of a person long jump.,a photo of a person table tennis shot."],
    ["demo/images/volleyball.png", "배구 스파이크하는 사람의 사진.,줄넘기하는 사람의 사진.,축구 페널티하는 사람의 사진.,멀리뛰기하는 사람의 사진.,탁구 치는 사람의 사진."],
    ]

    gr.Interface(classify,
                 inputs,
                 outputs,
                 title=title,
                 description=description,
                 examples=examples,
                 examples_per_page=50,
                 server_name="0.0.0.0",
                 server_port=10000
                 ).launch()


