import os
import platform
import torch
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import logging
import time

# Configure logging to write to a file
logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

tokenizer = AutoTokenizer.from_pretrained("/cpfs01/projects-SSD/cfff-5822cced0fd0_SSD/cxy_22210240138/PSG/PSG/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/cpfs01/projects-SSD/cfff-5822cced0fd0_SSD/cxy_22210240138/PSG/PSG/chatglm3-6b", trust_remote_code=True, device='cuda')
model = model.eval()

def gen_func_abs(func_name, doc_tem, visited, apk, start_time, time_limit):
    temp_result = {}
    if time.time() - start_time > time_limit:
        return func_name
    path = '/cpfs01/projects-SSD/cfff-5822cced0fd0_SSD/cxy_22210240138/PSG/PSG/PSG_dataset/describeCtx_function/'+apk+'/.'+func_name
    if os.path.exists(path):
        if func_name in doc_tem.keys():
            # print('$$$$$$$$$$$$$$$$$$$$$$')
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()
            code = code + '\n' + 'Below is a summary of the functionalities of the functions called in the above code segment:'
            func_names = doc_tem[func_name].split('||')
            for i in func_names:
                if os.path.exists('/cpfs01/projects-SSD/cfff-5822cced0fd0_SSD/cxy_22210240138/PSG/PSG/PSG_dataset/describeCtx_function/'+apk+'/.'+i):
                    if i not in visited:
                        visited.add(i)
                        # print('add code sub:'+i)
                        with open('/cpfs01/projects-SSD/cfff-5822cced0fd0_SSD/cxy_22210240138/PSG/PSG/PSG_dataset/describeCtx_function/'+apk+'/.'+i, 'r', encoding='utf-8') as f:
                            add_code = f.read()
                        code = code + '\n' + add_code
                        if i in doc_tem.keys():
                            sub_funcs = doc_tem[i].split('||')
                            for j in sub_funcs:
                                if time.time() - start_time > time_limit:
                                    break
                                if os.path.exists('/cpfs01/projects-SSD/cfff-5822cced0fd0_SSD/cxy_22210240138/PSG/PSG/PSG_dataset/describeCtx_function/'+apk+'/.'+j):
                                    if j not in visited:
                                        visited.add(j)
                                        # print('generate code sub'+i)
                                        temp_result[j] = gen_func_abs(i, doc_tem, visited, apk, start_time, time_limit)  
            for i in temp_result.keys():
                code = code + '\n' + temp_result[i]
            check = tokenizer.tokenize("code segment:\n"+code+"\nWhat is the purpose of the code segment?")
            if len(check)<7700:
                # print('genreate main code:'+i)
                response, history = model.chat(tokenizer, "code segment:\n"+code+"\nWhat is the purpose of the code segment?", history=[{'role':'user' ,'content':'code segment:\npublic void onSessionStateChange(Session session, SessionState state, Exception exception) {\n    if (state.isOpened()) {\n        Log.e("fb-login", "State opened");\n        facebookSuccess(session);\n    } else if (state.isClosed()) {\n        Log.e("fb-login", "State closed");\n    }\n}\nBelow is a summary of the functionalities of the functions called in the above code segment:\ncom.facebook.SessionState.isOpened(): It is a method associated with the state object that checks if the current session state is "opened". The state object is from the Facebook SDK and determines if the Facebook session is opened.\ncom.todoroo.astrid.ActFmLoginActivity.facebookSuccess(): It is a method that displays a progress dialog, sends a request to retrieve the user\'s information from Facebook, and handles the response. It extracts the user\'s email, first name, last name, and access token, then calls the authenticate() method with this information.\ncom.facebook.SessionState.isClosed(): It is a method associated with the state object that checks if the current session state is "closed". The state object is from the Facebook SDK and determines if the Facebook session is closed.\nWhat is the purpose of the code segment?'},{'role': 'assistant', 'metadata': '', 'content': 'The method serves the purpose of responding to changes in the Facebook session state. When the session state changes to "opened", it triggers the facebookSuccess() method to initiates the retrieval of user information from Facebook. Conversely, when the session state changes to "closed", it logs a message indicating that the session has been closed.'}])
                return response
            # else:
                # print('length max:'+i)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()
            check = tokenizer.tokenize("code segment:\n"+code+"\nWhat is the purpose of the code segment?")
            if len(check)<7700:
                #print('genreate code:'+func_name)
                response, history = model.chat(tokenizer, "code segment:\n"+code+"\nWhat is the purpose of the code segment?", history=[{'role':'user' ,'content':'code segment:\n@Override // android.content.DialogInterface.OnClickListener\npublic void onClick(DialogInterface dlg, int which) {\n    String firstName = firstNameField.getText().toString();\n    String lastName = lastNameField.getText().toString();\n    AndroidUtilities.hideSoftInputForViews(ActFmLoginActivity.this, firstNameField, lastNameField, email);\n    ActFmLoginActivity.this.authenticate(email.getText().toString(), firstName, lastName, ActFmInvoker.PROVIDER_PASSWORD, ActFmLoginActivity.this.generateRandomPassword());\n    StatisticsService.reportEvent(StatisticsConstants.ACTFM_SIGNUP_PW, new String[0]);\n}\nWhat is the purpose of the code segment?'},{'role': 'assistant', 'metadata': '', 'content': 'When the button in the dialog is clicked, a progress dialog will be created, and a series of operations will be executed in a new thread. These operations include rebuilding all synchronized data, completing the login operation on the UI thread, and finally closing the progress dialog.'}])
                return response
            else:
                return func_name
    else:
        # print(path)
        # print('genreate no exist:'+func_name)
        return func_name


def get_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirectories

# 指定目录的路径
directory_path = "/cpfs01/projects-SSD/cfff-5822cced0fd0_SSD/cxy_22210240138/PSG/PSG/PSG_dataset/describeCtx_fix/DescribeCTX_fix_result"

subdeirectories = get_subdirectories(directory_path)

subdeirectories = subdeirectories[90:]

time_limit_per_app = 20*60

app_processing_log = './describeCtx_time8.log'

for subdirectory in tqdm(subdeirectories):
    start_time_per_app = time.time()
    path = directory_path + "/" + subdirectory + "/"
    if os.path.exists(path+'uiHtml.json'):
        page_list = {}
        with open(path+'uiHtml.json', 'r', encoding='utf-8') as f:
            page_list = json.load(f)
        file_list = []
        for i in page_list.keys():
            file_list.append('/cpfs01/projects-SSD/cfff-5822cced0fd0_SSD/cxy_22210240138/PSG/PSG/PSG_dataset/describeCtx_template/'+subdirectory+'/'+i)
        page_length = len(file_list)
        for file in tqdm(file_list):
            start_time_per_function = time.time()
            doc_line = []
            try:
                with open(file,'r', encoding='utf-8') as f:
                    for line in f:
                        doc_line.append(line.strip())
            except Exception as e:
                # Handle the exception here
                error_message = f"Error processing doc_result[{i}]: {e}"
                # Log the error message to the file
                logging.error(error_message)
                # Optionally, print the error message to the console
                print(error_message)

            level = []
            func = []
            para = []

            for line in doc_line:
                # print(line)
                line = line.split('||')
                if line[-1]=='invoke':
                    level.append(line[1])
                    func.append(line[-3])
                    para.append(line[-2][1:].replace('$','.'))

            doc_template = {}
            current_branch = 'root'

            for i in range(len(level)):
                try:
                    if i>0:
                        if int(level[i])<int(level[i-1]):
                            for _ in range(int(level[i-1])-int(level[i])):
                                for key in doc_template.keys():
                                    backs = doc_template[key].split('||')
                                    flag = current_branch
                                    for back in backs:
                                        if current_branch == back:
                                            current_branch = key
                                            break
                                    if current_branch!=flag:
                                        break
                    if current_branch in doc_template.keys():
                        doc_template[current_branch] = doc_template[current_branch]+'||'+func[i] + '##' + para[i]
                    else:
                        doc_template[current_branch] = func[i] + '##' + para[i]
                    # print(current_branch,'||',func[i])
                    # input()
                    if i<len(level)-1:
                        if int(level[i+1])>int(level[i]):
                            current_branch = func[i] + '##' + para[i]
                except Exception as e:
                    # Handle the exception here
                    error_message = f"Error processing {file}]: {e}"
                    # Log the error message to the file
                    logging.error(error_message)
                    # Optionally, print the error message to the console
                    print(error_message)
                
            # print(doc_template)
            doc_result = {}
            # break
            for i in doc_template.keys():
                if i == 'root':
                    func = doc_template[i].split('||')
                    for j in func:
                        doc_result[j] = ''
                    break
            func_length = len(doc_result)
            page_per_time = time_limit_per_app/page_length
            if func_length>0:
                func_per_time = (page_per_time-2*func_length)/func_length
            else:
                func_per_time = page_per_time
            if func_per_time<0:
                func_per_time = 1
            for i in doc_result.keys():
                visited = set()
                visited.add(i)
                try:
                    doc_result[i] = gen_func_abs(i, doc_template, visited, subdirectory, time.time(), func_per_time)
                except Exception as e:
                    # Handle the exception here
                    error_message = f"Error processing doc_result[{i}]: {e}"
                    # Log the error message to the file
                    logging.error(error_message)
                    # Optionally, # print the error message to the console
                    print(error_message)
            if not os.path.exists('/cpfs01/projects-SSD/cfff-5822cced0fd0_SSD/cxy_22210240138/PSG/PSG/PSG_dataset/describeCtx_fix_abstract/'+subdirectory):
                os.makedirs('/cpfs01/projects-SSD/cfff-5822cced0fd0_SSD/cxy_22210240138/PSG/PSG/PSG_dataset/describeCtx_fix_abstract/'+subdirectory)
            with open('/cpfs01/projects-SSD/cfff-5822cced0fd0_SSD/cxy_22210240138/PSG/PSG/PSG_dataset/describeCtx_fix_abstract/'+subdirectory+'/'+file.replace('/cpfs01/projects-SSD/cfff-5822cced0fd0_SSD/cxy_22210240138/PSG/PSG/PSG_dataset/describeCtx_template/'+subdirectory+'/', ''), 'w', encoding='utf-8') as f:
                json.dump(doc_result, f)
    app_processing_time = time.time() - start_time_per_app
    with open(app_processing_log, 'a') as log_file:
        log_file.write(f'{subdirectory}: {app_processing_time} seconds\n')

        # com.app_reddamstaff.layout