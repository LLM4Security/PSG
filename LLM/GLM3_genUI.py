import os
import platform
import torch
from transformers import AutoTokenizer, AutoModel
import json

def get_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirectories

# 指定目录的路径
directory_path = "./AndroidBench_result"

tokenizer = AutoTokenizer.from_pretrained("/mnt/maldect_ssd_2T/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/mnt/maldect_ssd_2T/chatglm3-6b", trust_remote_code=True, device='cuda')
model = model.eval()

# 获取目录下的所有子目录
subdirectories = get_subdirectories(directory_path)

for subdirectory in subdirectories:
    path = directory_path + "/" + subdirectory + "/" + "uiHtml.json"
    if os.path.exists(path):
        html_result = {}
        with open(directory_path + "/" + subdirectory + "/" + "uiHtml.json", "r", encoding='utf-8') as f:
            print(subdirectory)
            uiHtml = json.load(f)
        for i in uiHtml.keys():
            response, history = model.chat(tokenizer, "Screen:\n"+uiHtml[i]+"\nNow reasoning starts:\nWhat is the purpose of the screen?", history=[{'role':'user' ,'content':'Screen:\n<p id=0 class="alertTitle"> Create password </p>\n<div id=1 class="titleDivider"> </div>\n<input id=2 class="password"> Crowd3116 </input>\n<input id=3 class="confirm password"> Crowd3116 </input>\n<input id=4 class="hint"> c3 </input>\n<input id=5 class="edEmailAddress"> appcrawler4@gmail.com </input>\n<p id=6 class="tvEmailAddressInfo"> This email address will be used to reset your password. </p>\n<button id=7 class="button2"> Cancel </button>\n<button id=8 class="button1"> 0K </button>\nNow reasoning starts:\nHow many input tags are there on the screen?'},{'role': 'assistant', 'metadata': '', 'content': '4'},{'role': 'user', 'content': 'What is the purpose of the screen?'},{'role': 'assistant', 'metadata': '', 'content': 'The screen is used to create a password'}])
            html_result[i] = response
        with open(directory_path + "/" + subdirectory + "/" + "ui_Abstract.json", "w") as f:
            json.dump(html_result, f)
# print(response)
# print(history)