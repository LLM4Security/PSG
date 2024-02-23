# %%
import os
import subprocess
import javalang

def get_method_start_end(method_node):
    startpos  = None
    endpos    = None
    startline = None
    endline   = None
    for path, node in tree:
        if startpos is not None and method_node not in path and isinstance(node, javalang.tree.MethodDeclaration):
            endpos = node.position
            endline = node.position.line if node.position is not None else None
            break
        if startpos is None and node == method_node:
            startpos = node.position
            startline = node.position.line if node.position is not None else None
    return startpos, endpos, startline, endline

def get_method_text(startpos, endpos, startline, endline, last_endline_index):
    if startpos is None:
        return "", None, None, None
    else:
        startline_index = startline - 1 
        endline_index = endline - 1 if endpos is not None else None 

        # 1. check for and fetch annotations
        if "@" in codelines[startline_index - 1]:
            startline_index = startline_index - 1

        meth_text = "<ST>".join(codelines[startline_index:endline_index])

        meth_text = meth_text[:meth_text.rfind("}") + 1] 

        # 2. remove trailing rbrace for last methods & any external content/comments
        # if endpos is None and 
        if not abs(meth_text.count("}") - meth_text.count("{")) == 0:
            # imbalanced braces
            brace_diff = abs(meth_text.count("}") - meth_text.count("{"))

            for _ in range(brace_diff):
                meth_text  = meth_text[:meth_text.rfind("}")]    
                meth_text  = meth_text[:meth_text.rfind("}") + 1]     
        
        meth_lines = meth_text.split("<ST>")  
        meth_text  = "".join(meth_lines)                   
        last_endline_index = startline_index + (len(meth_lines) - 1) 

        return meth_text, (startline_index + 1), (last_endline_index + 1), last_endline_index

# decompile single class from apk using jadx
# 使用jadx反编译单个class文件
def decompile_java(name, path, apk):
    if not os.path.exists(path):
        os.mkdir(path)
    try:
        subprocess.run(["jadx", "--single-class", name, "--single-class-output", path, apk],
                        stdout=subprocess.PIPE,universal_newlines=True)
    except Exception as e:
        print(e)
        # subprocess.run(['rm','-rf',app['apktool_res']])

# 解析.java文件，提取其内全部方法体
def parse_java_file(target_file):
    global tree, codelines
    if not os.path.exists(target_file):
        print("Decompile fail! No target file: ", target_file)
        return
    with open(target_file, 'r') as r:
        codelines = r.readlines()
        code_text = ''.join(codelines)

    lex = None
    tree = javalang.parse.parse(code_text)    
    methods = {}
    for _, method_node in tree.filter(javalang.tree.MethodDeclaration):
        startpos, endpos, startline, endline = get_method_start_end(method_node)
        method_text, startline, endline, lex = get_method_text(startpos, endpos, startline, endline, lex)
        methods[method_node.name] = method_text

    return methods

# 输入类名，提取该类的生命周期方法
def extract_lifecycle(name, javaFilePath, apkFilePath):
    res = {}
    jname = name.split(".")[-1].split("$")[0]
    target_file = f"{javaFilePath}/{jname}.java"
    if not os.path.exists(target_file):
        decompile_java(name, javaFilePath, apkFilePath)
        
    methods = parse_java_file(target_file)
    
    if not methods:
        return res
    
    for name, text in methods.items():
        if not text.splitlines():
            continue
        if " android." in text.splitlines()[0] and "@Override" in text.splitlines()[0]:
            res[name] = text
    return res

# 依据方法签名提取方法体源码
# 方法签名格式：<com.todoroo.astrid.reminders.NotificationFragment$SnoozeDialog: void onClick(android.content.DialogInterface,int)>
def get_method_text_from_signature(signature):
    if not signature:
        return False
    innerClass = signature[1:-1].split(": ")[0].replace("$", ".")
    sub_signature = signature[1:-1].split(": ")[1].split("(")[0]

    for ic in innerClassIndex:
        if innerClass in codelines[ic]:
            im = ic + 1
            annotation = ""
            if "@" in codelines[im]:
                annotation = codelines[im]
                im += 1
            if sub_signature in codelines[im]:
                method_text = ""
                for line in codelines[im:]:
                    method_text += line
                    if method_text.count("{") == method_text.count("}"):
                        method_text = annotation + method_text
                        return method_text
    return False

# 输入类名与回调方法签名，提取该类中回调方法
def extract_callbacks(callbacks, name, javaFilePath, apkFilePath):
    global innerClassIndex, codelines
    res = {}
    jname = name.split(".")[-1].split("$")[0]
    target_file = f"{javaFilePath}/{jname}.java"
    if not os.path.exists(target_file):
        decompile_java(name, javaFilePath, apkFilePath)
    if not os.path.exists(target_file):
        print("Decompile fail! No target file: ", target_file)
        return res
    with open(target_file, 'r') as r:
        codelines = r.readlines()

    innerClassIndex = []
    
    
    for index, line in enumerate(codelines):
        if "// from class: " in line:
            innerClassIndex.append(index)

    for signature in callbacks:
        method_text = get_method_text_from_signature(signature)
        if method_text:
            res[signature] = method_text
    return res
# %%
        
# %%

# %%
