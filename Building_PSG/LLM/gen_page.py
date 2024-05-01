import os
import platform
import torch
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import re

tokenizer = AutoTokenizer.from_pretrained("/home/chatglm3-6b32k", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/chatglm3-6b32k", trust_remote_code=True, device='cuda')
model = model.eval()



def get_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirectories

# 静态分析结果目录
directory_path = ""

# function summary结果目录
function_path = ""

# 输出目录
output_path = ""

# 获取目录下的所有子目录
subdirectories = get_subdirectories(directory_path)
for subdirectory in subdirectories:
    path = directory_path + "/" + subdirectory + "/"
    if os.path.exists(path+'uiHtml.json') and os.path.exists(path+'/cfgs'):
        ui_dic = {}
        with open(path+'uiHtml.json', 'r', encoding='utf-8') as f:
            ui_dic = json.load(f)

        ui_event = {}
        with open(path+'uiEvent.json', 'r', encoding='utf-8') as f:
            ui_event = json.load(f)

        path = function_path+subdirectory
        result = {}

        for i in tqdm(ui_dic.keys()):
            visited = set()
            page_title = i
            page_ui = ui_dic[i]
            page_event = ui_event[i]
            if os.path.exists(path+'/'+i):
                with open(path+'/'+i,'r', encoding='utf-8') as f:
                    page_func = json.load(f)
                    
                llm_input = f'Given a fused page content, you needs to generate a page summary that reflects the functionaity and intention of this page, by considering the semanticinformation fromh both code and UI.\nThis is a page called {page_title}, which has the following lifecycle events:\n'
                for _ in page_func.keys():
                    if i+'.onCreate' in _:
                        llm_input = llm_input + _ + ':' + page_func[_] + '\n'
                        visited.add(_)
                        break
                llm_input+='The following UI is loaded by onCreate:\n'
                llm_input+=page_ui
                llm_input+='\n'
                for _ in page_func.keys():
                    if i+'.onResume' in _:
                        llm_input = llm_input + _ + ':' + page_func[_] + '\n'
                        visited.add(_)
                        break
                llm_input+='After onResume, there are several events when user interacts with this page:\n'
                for _ in page_event.keys():
                    for event in page_event[_]:
                        pattern = r'<(.*)>'
                        tmp = re.findall(pattern, event)[0]
                        tmp = tmp.split(' ')
                        pattern = r'\((.*)\)'
                        para = re.findall(pattern, tmp[-1])[0]
                        tmp = tmp[0][:-1]+'.'+tmp[-1].split('(')[0]+'##'+para.replace(',', ', ')
                        if tmp in page_func.keys():
                            llm_input+=f'id={_} triggers the {event}:{page_func[tmp]}\n'
                        visited.add(tmp)
                llm_input+='\nAdditionally, this page may execute the following events that are transparent to user:\n'
                final_input = ''
                for _ in page_func.keys():
                    if i+'.onPause' in _:
                        final_input = final_input + _ + ':' + page_func[_] + '\n'
                        visited.add(_)
                        break

                for _ in page_func.keys():
                    if i+'.onStop' in _:
                        final_input = final_input + _ + ':' + page_func[_] + '\n'
                        visited.add(_)
                        break
                
                for _ in page_func.keys():
                    if i+'.onDestroy' in _:
                        final_input = final_input + _ + ':' + page_func[_] + '\n'
                        visited.add(_)
                        break

                for _ in page_func.keys():
                    if _ not in visited:
                        visited.add(_)
                        if _ != page_func[_]:
                            llm_input = llm_input + _ + ':' + page_func[_] + '\n'
                llm_input+=final_input
                # print(llm_input)
                check = tokenizer.tokenize(llm_input)
                print(len(check))
                response, history = model.chat(tokenizer, llm_input+"\nConside the semantic information from both the code and Ul to generate a page summary that reflects the functionality and intentions of this page.?", history=[{'role':'user' ,'content':'Given a fused page content, you needs to generate a page summary that reflects the functionaity and intention of this page, by considering the semanticinformation fromh both code and UI.\n This is a page called \{com.todoroo.astrid.actfm.ActFmLoginActivity\}, which has the following lifecycle events:\ncom.todoroo.astrid.actfm.ActFmLoginActivity.onCreate##android.os.Bundle: The purpose of this code segment is to initialize the UI of the FragmentActivity and set up the necessary components, such as the title, action bar, and page view. It also sets up a random number generator and a UiLifecycleHelper to manage the lifecycle of the UI components.\nThe following UI is loaded by onCreate:\n<div id=0 class="pager"> </div>\n<div id=1 class="indicator"> </div>\n<p id=2 class="next"> Next </p>\n<img id=3 class="logo"> </img>\n<p id=4 class="error"> </p>\n<button id=5 class="fb_login_dummy"> Connect with Facebook </button>\n<button id=6 class="gg_login"> Connect with Google </button>\n<p id=7 class="pw_login"> Log in </p>\n<p id=8 class="pw_signup"> Sign up </p>\ncom.todoroo.astrid.actfm.ActFmLoginActivity.onResume##: The code segment is a method that is called when the app is resumed after being paused or stopped. It is used to perform any necessary initialization or setup tasks that need to be completed before the app can be fully functional. In this case, it starts the session of the StatisticsService.\nAfter onResume, there are several events when user interacts with this page:\n{id=5} triggers the {<com.todoroo.astrid.actfm.ActFmLoginActivity$4: void onClick(android.view.View)>} function: The purpose of the code segment is to get the current Facebook session. It checks for the existence of an active session and returns it if it exists. If an active session does not exist, the method will block until an active session is created. This method is used by other parts of the code to check for an active session and perform actions accordingly.\n{id=6} triggers the {<com.todoroo.astrid.actfm.ActFmLoginActivity$3: void onClick(android.view.View)>} function: The purpose of the given code segment is to start the Google Auth activity when a button is clicked. The code creates an Intent that starts the Google Auth activity and starts the activity using the startActivityForResult() method. The second parameter of the startActivityForResult() method is an integer that represents the result code for the activity. The code also calls the reportEvent() method of the StatisticsService to record an event related to the login process.\n{id=7} triggers the {<com.todoroo.astrid.actfm.ActFmLoginActivity$6: void onClick(android.view.View)>} function: The purpose of the given code segment is to define an OnClickListener for a sign-in button in the ActFmLoginActivity class. When the sign-in button is clicked, the onClick method of the OnClickListener is executed.Inside the onClick method, an AlertDialog is created and displayed, which contains a form for the user to enter their email and password. It also includes a "Forgot password" link. The user can enter their email and password, and then choose to either authenticate with the provided credentials or cancel the sign-in process. Overall, the code segment handles the UI interaction for the sign-in functionality in the ActFmLoginActivity class.\n{id=8} triggers the {<com.todoroo.astrid.actfm.ActFmLoginActivity$5: void onClick(android.view.View)>} function: The purpose of the code segment is to handle the user authentication process for the Facebook login. It sets up the UI for the user to enter their Facebook login information such as their first name, last name, and email address. Once the user clicks the \"Sign Up\" button, an Android AlertDialog is displayed with two buttons - \"Sign Up\" and \"Cancel\". The \"Sign Up\" button initiates a request to authenticate the user\'s information using the Facebook SDK, and if successful, the user\'s information is passed to the authenticate() method including a random password generated locally.\nAdditionally, this page may execute the following events that are transparent to user:\ncom.todoroo.astrid.actfm.ActFmLoginActivity.onActivityResult##int, int, android.content.Intent: This code segment is an `onActivityResult()` method in the `FragmentActivity` class. It is called when the result of an activity\'s `requestCode` is available. The method checks the `requestCode` and `resultCode` to determine the result of the activity\'s operation. If the `resultCode` is not 0, it checks the result of the operation and performs specific actions based on the data received in the `data` parameter. In this case, it either calls the `credentialsListener.getCredentials()` method or the `authenticate()` method depending on the data received in the `data` parameter.\ncom.todoroo.astrid.actfm.ActFmLoginActivity.onConfigurationChanged##android.content.res.Configuration: The purpose of the code segment is to set the content view of the SherlockFragmentActivity to the content view resource specified in the AndroidManifest.xml file.\nAnd other lifecycle events:\ncom.todoroo.astrid.actfm.ActFmLoginActivity.onPause##: This method is called when the activity is paused, either by the user pressing the home button or by the system when it is being killed to free up memory. In this method, the code is pausing the activity by calling the `onPause()` method of its superclass, and also calling the `onPause()` method of the `uiHelper` and `StatisticsService`. This will pause the activity and any ongoing operations, and prepare it for resuming when the user is ready to use it again.\ncom.todoroo.astrid.actfm.ActFmLoginActivity.onStop##: This code segment is responsible for stopping the `StatisticsService` when the `SherlockFragmentActivity` is stopped. It calls the `sessionStop()` method of the `StatisticsService` and then calls the `super.onStop()` method to ensure that any necessary cleanup operations are performed.\ncom.todoroo.astrid.actfm.ActFmLoginActivity.onDestroy##: The `onDestroy()` method is typically called when an activity is about to be destroyed. In this case, it is being called by the `SherlockFragmentActivity` class, which is likely a custom class that extends `FragmentActivity` or `AppCompatActivity`.\n\nInside this method, the `super.onDestroy()` call is used to call the `onDestroy()` method of the parent class, which is expected to be implemented by any custom class that extends `Activity`. This is a common practice in Android development to ensure that any custom class that extends `Activity` properly cleans up any resources it has created before the app is destroyed.\n\nThe `this.uiHelper.onDestroy()` call is likely used to destroy any resources that the `uiHelper` object has created. The `uiHelper` object is likely an instance of a utility class that provides helper methods for the activity, such as handling views, data binding, or other UI-related tasks.\n\nOverall, this code segment is ensuring that any necessary cleanup is performed before the activity is destroyed, and any resources that were created by the activity are properly released.\nGive a description on what this page do by considering its title and lifecycle functions?'},{'role': 'assistant', 'metadata': '', 'content': 'This Activity is for login and authentication. It starts the session of StatisticsService in onResume(), likely for tracking user statistics or analytics. It also processes results from other activities, executing specific actions based on received data in onActivityResult(), such as invoking credentialsListener.getCredentials() or authenticate().'},{'role':'user' ,'content':'List all the UI elements on the page.'},{'role': 'assistant', 'metadata': '', 'content': '<div id=0 class="pager"> </div>\n<div id=1 class="indicator"> </div>\n<p id=2 class="next"> Next </p>\n<img id=3 class="logo"> </img>\n<p id=4 class="error"> </p>\n<button id=5 class="fb_login_dummy"> Connect with Facebook </button>\n<button id=6 class="gg_login"> Connect with Google </button>\n<p id=7 class="pw_login"> Log in </p>\n<p id=8 class="pw_signup"> Sign up </p>'},{'role':'user' ,'content':'List all the UI elements that interact with users and summarize what they do by considering the text and code bound to these elements.'},{'role': 'assistant', 'metadata': '', 'content': '1. id = 5 implements Connect with Facebook: This interaction allows the user to log in or authenticate using their Facebook account.\n2. id = 6 implements Connect with Google: This interaction enables the user to log in or authenticate using their Google account.\n3. id = 7 implements Log in: This interaction provides a form for the user to enter their email and password for authentication.\n4. id = 8 implements Sign up: This interaction allows the user to sign up with their Facebook account and authenticate their information.'},{'role':'user' ,'content':'List the sensitive behaviors (if any) by considering the whole page content.'},{'role': 'assistant', 'metadata': '', 'content': '1. Random password generation: The code segment related to user authentication with Facebook includes the generation of a random password locally. It suggests that the page may be storing or transmitting sensitive user credentials. \n2. Statistics tracking: The StatisticsService, mentioned in the onResume() method, indicates that user events or statistics are being tracked.'},{'role':'user' ,'content':'Conside the semantic information from both the code and Ul to generate a page summary that reflects the functionality and intentions of this page.'},{'role': 'assistant', 'metadata': '', 'content': 'The purpose of this page is to provide a login and authentication functionality to users.\nThe user interface (UI) of the page includes options to connect with Facebook and Google for convenient login options. It also allows users to manually log in with their email and password or sign up with their Facebook account.\nFurthermore, this page performs some privacy-related actions that are transparent to users, such as storing or transmitting sensitive user password during login, and tracking user events or statistics.'}])
                print(response)
                result[i] = response
            if not os.path.exists(output_path+subdirectory):
                os.makedirs(output_path+subdirectory)
            with open(output_path+subdirectory+'/page.json','w', encoding='utf-8') as f:
                json.dump(result,f)
