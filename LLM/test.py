
# doc_line = []

# with open('/home/jiuyan/PPG/astrid_test/astrid_template/com.facebook.LoginActivity','r') as f:
#     for line in f:
#         doc_line.append(line.strip())

# level = []
# func = []

# for line in doc_line:
#     line = line.split('||')
#     if line[-1]=='invoke':
#         level.append(line[1])
#         func.append(line[-2])

# tem = {}
# current_branch = 'root'

# for i in range(len(level)):
#     if i>0:
#         if int(level[i])<int(level[i-1]):
#             for _ in range(int(level[i-1])-int(level[i])):
#                 for key in tem.keys():
#                     if current_branch in tem[key]:
#                         current_branch = key
#                         break
#     if current_branch in tem.keys():
#         tem[current_branch] = tem[current_branch]+'||'+func[i]
#     else:
#         tem[current_branch] = func[i]
#     if i<len(level)-1:
#         if int(level[i+1])>int(level[i]):
#             current_branch = func[i]

# print(func)
# print(level)

# print(tem)

import json
with open('/home/jiuyan/PPG/AndroidBench_page_ori/astrid/page.json', 'r', encoding='utf-8') as f:
    page_dic = json.load(f)

for page in page_dic:
    print(page)