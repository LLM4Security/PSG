
def any_tree_to_html(node):
  """Turns an AnyTree representation of view hierarchy into HTML.

  Args:
    node: an AnyTree node.
    layer: which layer is the node in.

  Returns:
    results: output HTML.
  """
  results = ''
  if not "type" in node:
    return results
  
  if 'IMAGEVIEW' == node["type"]:
    node_type = 'img'
  elif 'BUTTON' == node["type"]:
    node_type = 'button'
  elif 'EDITTEXT' == node["type"]:
    node_type = 'input'
  elif 'TEXTVIEW' == node["type"]:
    node_type = 'p'
  else:
    node_type = 'div'
  if node["is_leaf"] and node["visible"]:
    html_close_tag = node_type
    results = '<{}{}{}{}{}{}> {} </{}>\n'.format(
        node_type,
        ' id={}'.format(node["leaf_id"]) if node["leaf_id"] != -1 else '',
        ' class="{}"'.format(node["resource_id"]) if node["resource_id"] else '',
        ' type="{}"'.format(node["inputType"]) if node["inputType"] else '',
        ' alt="{}"'.format(node["contentDescription"]) if "contentDescription" in node else '',
        ' placeholder="{}"'.format(node["hint"]) if "hint" in node else '',
        '{}'.format(node["text"]) if "text" in node else '',
        html_close_tag,
    )
  else:
    children_results = ''
    for child in node["children"]:
      children_results += any_tree_to_html(child)
    results += children_results

  return results

             
def parse_layout_json(obj, leaf_count):
  """Parses a layout json string.

  Args:
    layout_json_string: the layout json string.

  Returns:
    page_html: html representation of the ui hierarchy in the page.
  """
  if "fragmentClass" in obj:
    return {}
      
  node_type = obj["viewClass"].split(".")[-1].upper()
  is_leaf = "children" not in obj or len(obj["children"]) == 0
  # 0: visible显示; 1: invisible显示黑背景条; 2: gone不显示
  visibility = obj.get("otherAttributes", {}).get("visibility", 0)
  visible = True if visibility == 0 else False
  clickable = obj.get("otherAttributes", {}).get("clickable", False)
  resource_id = obj.get("idVariable", "")
  inputType = obj.get("inputTypes", [""])[0]
  guid = obj.get("guid", "")
    
  node = {
    "type": node_type,
    "is_leaf": is_leaf,
    "clickable": clickable,
    # "scrollable": scrollable,
    # "checkable": checkable,
    "visible": visible,
    "leaf_id": len(leaf_count) if is_leaf and visible else -1,
    "resource_id": resource_id,
    "inputType": inputType,
    "children": []
  }
  
  textAttributes = obj.get("textAttributes", [])
  for textAttribute in textAttributes:
    if not isinstance(textAttribute, dict):
      continue
    node[textAttribute.get("name", "text")] = textAttribute.get("value", textAttribute.get("variable", "")) 
  
  if is_leaf and visible:
    leaf_count.append(guid) 
    
  children = obj.get("children", [])
  for child in children:
    node["children"].append(parse_layout_json(child, leaf_count))

  return node

def parse_dialog_json(obj, leaf_count):
  """Parses a dialog json string.

  Args:
    dialog_json_string: the dialog json string.

  Returns:
    dialog_html: html representation of the ui hierarchy in the dialog.
  """
  node_type = 'dialog'
  
  html_close_tag = node_type
  results = '<{}>'.format(node_type)
  
  for title in obj.get('titles', []):
    results += ' <h2>{}</h2>'.format(title)
    
  for message in obj.get('messages', []):
    results += ' <p>{}</p>'.format(message)
  
  for button in obj.get('buttons', []):
    results += ' <button id={}>{}</button>'.format(len(leaf_count), button.get('label', ''))
    leaf_count.append(button.get('guid', ''))
        
  results += ' </{}>\n'.format(html_close_tag)

  return results

# %%
