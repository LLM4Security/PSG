# %%
from collections import defaultdict
import json
import operator
import os
import re
from tfidf import generate_tfidf_vectors
import pandas as pd

def evaluate_security(graph, login_credentials, recovery_credentials):
    has_backdoor = False
    has_invalid_recovery = False

    # Check if security is compromised by the addition of recovery process
    if len(recovery_credentials) > 0:
        has_backdoor = True

    # Check if there are invalid recovery mechanisms (isolated recovery credentials)
    for credential in recovery_credentials:
        if graph.degree(credential) == 0:
            has_invalid_recovery = True
            break

    return has_backdoor, has_invalid_recovery

def generate_inverse_index(tfidf_dict):
    inverse_index = defaultdict(list)

    # 构建倒排索引
    for screen_id, tfidf_vector in tfidf_dict.items():
        for term, tfidf_score in tfidf_vector.items():
            inverse_index[term].append((screen_id, tfidf_score))

    return inverse_index

def search_pages(inverse_index, search_term):
    # 搜索包含搜索词的页面
    search_results = inverse_index.get(search_term, [])

    # 根据TF-IDF分数对搜索结果进行排序
    sorted_results = sorted(search_results, key=operator.itemgetter(1), reverse=True)

    return sorted_results

def search_page(tfidf_dict, search_term):
    
    print(f"-搜索标签 \'{search_term}\' 中...")
    
    # 生成倒排索引
    inverse_index = generate_inverse_index(tfidf_dict)

    # 搜索包含搜索词的页面并按TF-IDF分数排序
    results = search_pages(inverse_index, search_term)

    print(f"-结果：")

    # 打印搜索结果
    for page_id, tfidf_score in results:
        print(f"页面: {page_id}, TF-IDF分数: {tfidf_score}")
    
    page_ids = [page_id for page_id, tfidf_score in results] 
    return page_ids
       
# %%
def generate_data(abstracts_file, tfidf_file):
    
    generate_tfidf_vectors(abstracts_file, tfidf_file)

    # sensitive_behavior = {
    #     'contact': ['ringtone', 'contact'],
    #     'microphone': ['microphone'],
    #     'storage': ['device storage', 'external storage'],
    #     'camera': ['camera'],
    #     'calendar': ['calendar', 'reminder', 'schedule'],
    #     'SMS': ['SMS', 'send message', 'text message'],
    #     'location': ['location', 'gps', 'nearest', 'nearby']
    # }
    
    sensitive_behavior = {
        'contact': ['address book', 'phone book','assign','ringtone','contacts'],						  
        'microphone': ['microphone', 'record audio','voice','measure','speak','recognition'],						  
        'location': ['location', 'gps','track','nearest','nearby','geographical location',' geo-location'],						  
        'storage':['device storage','access external storage','download','external storage','save'],						  'camera':['camera','take photo','take picture','record video', 'video recording'],
        'calendar':['calendar access','event','calendar','reminder','calendar permission','access calendar','personal calendar','schedule'],
        'SMS':['SMS',"send message",'text message']
	}

    with open(abstracts_file) as file:
        abstracts = json.load(file)    

    # 从JSON文件加载TF-IDF向量
    with open(tfidf_file) as file:
        tfidf_dict = json.load(file)

    sen_pages = {}
    for permission in sensitive_behavior:
        sen_pages[permission] = search_page(tfidf_dict, permission)

    # sensitive_description = {}
    sensitive_description = {'contact': [], 'microphone': [], 'location': [], 'storage': [], 'camera': [], 'calendar': [], 'SMS': []}
    for permission in sen_pages:
        for page in sen_pages[permission]:
            # 提取页面的文本内容
            summary = abstracts[page]
            sentences = re.findall(r'[\w\s]+', summary)

            for text in sentences:
                text = text.lower()
                for behavior in sensitive_behavior[permission]:
                    if behavior in text and len(sensitive_description[permission]) < 3:
                        sensitive_description[permission].append(text.lower().replace('\n', ' '))
       
            break
    
    return sensitive_description
# %%
result_path = "/mnt/iscsi/cqt/ppg/DescribeCTX_result"
resList = [folder for folder in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, folder))]

dataset = pd.read_csv("/mnt/iscsi/cqt/DescribeCTX/Data/dataset_1262.csv")
data = []

for index, row in dataset.iterrows():
    name = str(row['AppName'])
    des = str(row['OriginalDescription'])
    category = str(row['Permission'])
    abstracts_file = os.path.join(result_path, name, "page_abstract.json")
    # print(abstracts_file)
    if os.path.exists(abstracts_file):
        # os.mkdir(os.path.join(tfidf, app))
        tfidf_file = os.path.join(result_path, name, "tfidf_vectors.json")
        sensitive_description = generate_data(abstracts_file, tfidf_file)
        
        for permission in sensitive_description:
            if sensitive_description[permission] != []:
                summary = ' '.join(sensitive_description[permission])
                data.append({'apk': name, 'sum': summary, 'per': permission, 'des': des}) 
        # print(sensitive_description)

data = pd.DataFrame(data)

data.to_csv("/mnt/iscsi/cqt/DescribeCTX/Code/data_psg.csv", index=False)  
# %%
# data = data.drop(data[data['sum'].str.len() > 1000].index)

# data.to_csv("/mnt/iscsi/cqt/DescribeCTX/Code/data_psg.csv", index=False)  
# %%

# %%
