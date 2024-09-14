# %%
# -*- coding: utf-8 -*-
import glob
import os
import json
import shutil
from multiprocessing import Pool, TimeoutError
import subprocess
import time
from ui2html import parse_dialog_json, parse_layout_json, any_tree_to_html
def parse_page(page, leaf_count):

  # === 1. 提取UI布局信息，转换成html格式 ===
  uiHtml = ""

  idx = 0
  layouts = page.get("layouts", [])
  dialogs = page.get("dialogs", [])
  fragments = page.get("fragments", [])
  listeners = page.get("listeners", [])

  for obj in layouts:
    node = parse_layout_json(obj, leaf_count)
    page = any_tree_to_html(node)
    uiHtml += page
    idx += 1
  
  for f in fragments:
      uiHtml += parse_page(f, leaf_count)[0]
      
  for d in dialogs:
      uiHtml += parse_dialog_json(d, leaf_count)

  # === 2. 提取监听器信息，对应到UI的id ===
  callbacks = {}
  for l in listeners:
      id = l.get('guid', '')  
      if id in leaf_count:
          callbacks[leaf_count.index(id)] = [l['listener']] if 'listener' in l else l.get("listeners", [])
#   print(callbacks)
#     callbacks.append(l.get("listener", ""))
#     callbacks.extend(l.get("listeners", []))
  
  return uiHtml, callbacks



def extract_fragment_listener(node):
  # print(node["id"])
  fragments = []
  listeners = []
  if "fragmentClass" in node:
    # print(node["fragmentClass"])
    # 处理fragment内部的listener
    flisteners = []
    for layout in node.get("layouts", []):
      fl = extract_fragment_listener(layout)[1]
      flisteners.extend(fl)
    node['listeners'] = flisteners
    fragments.append(node)
  elif "listeners" in node and node["listeners"]:
    listeners.append(node)

  for child in node.get("children", []):
    f, l = extract_fragment_listener(child)
    fragments.extend(f)
    listeners.extend(l)

  return fragments, listeners

# %%
def build_basicPTG(result_path, apk):
  global javaFilePath, apkFilePath, codeJava, uiHtml
  
  print(f"build_basicPTG for {apk}...")
  
  if not os.path.exists(f'{result_path}/{apk}.json'):
    print(f"No json file {result_path}/{apk}.json")
    return
  
  with open(f'{result_path}/{apk}.json') as file:
    layout_tree = json.load(file)

  javaFilePath = f"{result_path}/java"
  apkFilePath = f"{result_path}/{apk}.apk"

  uiEvent = {}
  uiHtml = {}

  for a in layout_tree.get("activities", []):
      dialogs = a.get("dialogs", [])
      fragments = []
      listeners = []
      for layout in a.get("layouts", []):
          f, l = extract_fragment_listener(layout)
          fragments.extend(f)
          listeners.extend(l)
      for d in dialogs:
          for button in d.get("buttons", []):
              if "listener" in button and button["listener"]:
                  listeners.append(button)
      a['fragments'] = fragments
      a['listeners'] = listeners

  with open(f"{result_path}/ptg.json",'w')as nf:
      json.dump(layout_tree,nf,ensure_ascii=False)

  for a in layout_tree.get("activities", []):
      leaf_count = []
      act_name = a['name']
      uiHtml[act_name], uiEvent[act_name] = parse_page(a, leaf_count)
      
  # result_path = f"/mnt/iscsi/cqt/ppg/DeUEDroid_fix_result/Porn/{apk}"
  os.makedirs(result_path, exist_ok=True)
  # 保存html UI结果
  with open(f"{result_path}/uiHtml.json",'w')as nf:
      json.dump(uiHtml,nf,ensure_ascii=False)
  
  # 保存callback结果
  with open(f"{result_path}/uiEvent.json",'w')as nf:
      json.dump(uiEvent,nf,ensure_ascii=False)
      
  print(f"building success for {apk}...")

def static_analysic(result_path, apk_path, apk_name):
  if not os.path.exists(result_path):
    os.mkdir(result_path)
  # else:
    # print(result_path)
    # return
  
  print(" ".join(['java', '-jar', './static_analysis/ppg_sa.jar' , apk_path, apk_name]))
  
  try:
    subprocess.run(['java', '-jar', '-Xms10g', '-Xmx10g', './static_analysis/ppg_sa.jar' , apk_path, apk_name], timeout=1800, cwd = result_path, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  except Exception as e:
    print(f"Error {e} for {apk_name}...")
  
    
def copy_abstract2label(sc_path, res_path, apk_name):
  src = os.path.join(sc_path, "page.json")
  tgt = os.path.join(res_path, "page_abstract.json")
  label = os.path.join(res_path, "page_label.json")
  
  if os.path.exists(src):
    cmd = 'cp %s %s' % (src, tgt)
    os.popen(cmd)
    generate_tfidf_vectors(src, label)


if __name__ == "__main__":
    
  source_path = ""
  result_path = ""
  #`source_path`: Path where the APK files are stored
  #`result_path`: Output path for analysis results

  resList = [app[:-4] for app in os.listdir(source_path)]
  resList = [_.strip() for _ in resList]

  print(resList)
  
  start_time = time.time()

  p = Pool(16)
  for apk_name in resList:
      # print(apk_name)
      apk_path = os.path.join(source_path, apk_name+".apk")
      print(apk_path)
      res_path = f"{result_path}/{apk_name}"
      p.apply_async(static_analysic, args = (res_path, apk_path, apk_name))
  p.close()
  p.join()
  
  end_time = time.time()
  execution_time = end_time - start_time
  print("Execution Time (Process ", os.getpid(), "): ", execution_time, " seconds")
  
  p = Pool(16)
  for apk_name in resList:
      res_path = f"{result_path}/{apk_name}"
      p.apply_async(build_basicPTG,args=(res_path, apk_name))
  p.close()
  p.join()
  
# %%
