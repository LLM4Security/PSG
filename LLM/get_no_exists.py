import os 

def get_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirectories

subdirectories = get_subdirectories('./describeCtx_page')

apk_name = []
with open('./describeCtx.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        apk_name.append(line.strip())

not_gen_apk = []

for i in apk_name:
    if i not in subdirectories:
        not_gen_apk.append(i)

with open('./reamain_describeCtx.txt','w') as f:
    for i in not_gen_apk:
        f.write(i+'\n')