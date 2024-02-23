import os
import platform
import torch
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm

def get_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirectories

# 指定目录的路径
directory_path = "./astrid_test"

tokenizer = AutoTokenizer.from_pretrained("/mnt/maldect_ssd_2T/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/mnt/maldect_ssd_2T/chatglm3-6b", trust_remote_code=True, device='cuda')
model = model.eval()

# 获取目录下的所有子目录
subdirectories = get_subdirectories(directory_path)

for subdirectory in tqdm(subdirectories):
    file_list = os.listdir(directory_path + "/" + subdirectory)
    for i in tqdm(file_list):
        path = directory_path + '/' + subdirectory + '/' + i
        if os.path.exists(path):
            if 'com.todoroo.astrid' in path:
                with open(path, 'r') as f:
                    code = f.read()
                check = tokenizer.tokenize(code)
                if len(check)<8000:
                    response, history = model.chat(tokenizer, "code segment:\n"+code+"\nWhat is the purpose of the code segment?", history=[{'role':'user' ,'content':'code segment:\n@Override // android.content.DialogInterface.OnClickListener\npublic void onClick(DialogInterface dlg, int which) {\n    String firstName = firstNameField.getText().toString();\n    String lastName = lastNameField.getText().toString();\n    AndroidUtilities.hideSoftInputForViews(ActFmLoginActivity.this, firstNameField, lastNameField, email);\n    ActFmLoginActivity.this.authenticate(email.getText().toString(), firstName, lastName, ActFmInvoker.PROVIDER_PASSWORD, ActFmLoginActivity.this.generateRandomPassword());\n    StatisticsService.reportEvent(StatisticsConstants.ACTFM_SIGNUP_PW, new String[0]);\n}\nWhat is the purpose of the code segment?'},{'role': 'assistant', 'metadata': '', 'content': 'When the button in the dialog is clicked, a progress dialog will be created, and a series of operations will be executed in a new thread. These operations include rebuilding all synchronized data, completing the login operation on the UI thread, and finally closing the progress dialog.'}])
                    with open(directory_path+'/'+subdirectory+'/function_abstract/'+i, "w") as f:
                        json.dump(response, f)
