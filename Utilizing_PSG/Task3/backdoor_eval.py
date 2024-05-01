# %%
from collections import defaultdict
import json
import operator
import os
from tfidf import generate_tfidf_vectors

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
def generate_uaag(abstracts_file, tfidf_file):
    
    generate_tfidf_vectors(abstracts_file, tfidf_file)

    creWord = ["password", "sms", "email", "authenticator", "qr", "phone"]
    recWord = ["reset", "forget", "recover", "retrieve", "modify", "change"]

    with open(abstracts_file) as file:
        abstracts = json.load(file)    

    # 从JSON文件加载TF-IDF向量
    with open(tfidf_file) as file:
        tfidf_dict = json.load(file)

    login_pages = search_page(tfidf_dict, "login")

    credential_pages = list()
    fallback_pages = list()
    uaag = {page:{"account":[]} for page in login_pages}

    for credential in creWord:
        cred_pages = search_page(tfidf_dict, credential)
        
        # 收集提及多个凭证的页面
        for page in cred_pages:
            if page in credential_pages:
                fallback_pages.append(page)
            else:
                credential_pages.append(page)
                
            if page in uaag:
                uaag[page]["account"].append(credential)
    
    # security_socre(primary)
    # recoverability_score(primary)

    for page in fallback_pages:
        # 提取页面的文本内容
        text = abstracts[page]
        if not page in uaag:
            uaag[page] = {}
        controller = set()
        
        for recovery in recWord:
            for credential in creWord:
                if f"{recovery} {credential}" in text:
                    uaag[page][credential] = []
                    
            for credential in creWord:
                if credential in text and not credential in uaag[page]:
                    controller.add(credential)
        
        for cotrollee in uaag[page]:
            for credential in controller:
                if not credential == cotrollee and not credential in uaag[page][cotrollee]:
                    uaag[page][cotrollee].append(credential)
    return uaag
# %%
result_path = "/mnt/iscsi/cqt/ppg/backdoor_page"
resList = [folder for folder in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, folder))]

uaag = "/mnt/iscsi/cqt/ppg/uaag"

for app in resList:
    abstracts_file = os.path.join(result_path, app, "page.json")
    print(abstracts_file)
    if os.path.exists(abstracts_file):
        # os.mkdir(os.path.join(tfidf, app))
        tfidf_file = os.path.join(result_path, app, "tfidf_vectors.json")
        # generate_tfidf_vectors(abstracts_file, tfidf_file)
        os.mkdir(os.path.join(uaag, app))
        
        with open(os.path.join(uaag, app, "uaag.json"),'w')as nf:
            json.dump(generate_uaag(abstracts_file, tfidf_file),nf,ensure_ascii=False)
        print("success for {}".format(app))
# %%
import pandas as pd
topList = open("/mnt/iscsi/cqt/ppg/top500_round2").readlines()
topList = [_.strip() for _ in topList]
resList = open("/mnt/iscsi/cqt/ppg/top500_round3").readlines()
resList = [_.strip() for _ in resList]
# resList = pd.read_csv("/mnt/iscsi/cqt/2FA_result_new/output/results.csv")
for app in resList:
    if not app in topList:
        print(app)
# %%
