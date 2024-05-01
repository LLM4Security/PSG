import re
import os
import logging

# Configure logging to write to a file
logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

def parse_dot(dot_graph):
    graph = {}
    node_info = {}
    edge_info = {}

    lines = dot_graph.split("\n")
    for line in lines:
        pattern = re.compile(r'\[(.*)\]')
        matches = pattern.findall(line)
        if "->" in line and "->" not in matches[0]:
            parts = line.strip().split()
            source_node, target_node = parts[0], parts[2]
            edge_info[(source_node, target_node)] = line.strip()

            if source_node not in graph:
                graph[source_node] = []
            graph[source_node].append(target_node)

        elif "[" in line and "]" in line:
            pattern = re.compile(r'\[(.*)\]')
            matches = pattern.findall(line)
            node_parts = line.strip().split("[")
            node_id = node_parts[0].strip()
            node_info[node_id] = matches[0]

    return graph, node_info, edge_info

def custom_sort_key(item):
    order = ['inter', 'if', 'intra']
    return order.index(item[0])

def dfs(graph, current_node, visited, node_info, edge_info, output_file, tab_num):
    if current_node not in visited:
        visited.add(current_node)
        node = node_info.get(current_node, "")
        node = node.split('||')
        level = node[-1]
        node = node[0]
        output_file.write("{}||".format(level[:-2].split(' ')[-1]))
        if 'invoke' in node:
            if 'label="if' in node:
                output_file.write("{}|| ||else\n".format(node))
            else:
                tmp = node.split('label=')
                pattern = r'<(.*)>'
                doc_line = re.findall(pattern, tmp[1])
                pattern = r'\((.*)\)'
                para = re.findall(pattern, doc_line[0])
                para = para[0]
                doc_line = doc_line[0].split(' ')
                line_result = doc_line[0][:-1] + '.' + doc_line[-1].split('(')[0] + '|| ' + para.replace(',',', ')
                output_file.write("{}||invoke\n".format(line_result))
        else:
            output_file.write("{}||else\n".format(node))
        
        if current_node in graph:
            neighbor_node = []
            for neighbor in graph[current_node]:
                edge = edge_info.get((current_node, neighbor), "No info available")
                neighbor_node.append([edge.split('"')[1],neighbor])
            # print('####',neighbor_node)
            neighbor_node = sorted(neighbor_node, key=custom_sort_key)
            for i in neighbor_node:
                output_file.write("{}||".format(i[0])) 
                dfs(graph, i[1], visited, node_info, edge_info, output_file, tab_num)

def dfs_traversal(graph, start_node, node_info, edge_info, output_file):
    visited = set()
    dfs(graph, start_node, visited, node_info, edge_info, output_file, 0)


def get_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirectories

# 静态分析结果目录
directory_path = ""

# 结果输出目录
output_path = ""

# 获取目录下的所有子目录
subdirectories = get_subdirectories(directory_path)


for subdirectory in subdirectories:
    path = directory_path + '/' + subdirectory
    subsubdirectories = get_subdirectories(path)
    if os.path.exists(path):
        path = directory_path + "/" + subdirectory + "/" + "cfgs/dummyMainClass"
        if os.path.exists(path):
            file_list = os.listdir(path)
            for i in file_list:
                with open(path + '/' + i,'r') as f:
                    dot_graph = f.read()
                if not os.path.exists(output_path+subdirectory):
                    os.makedirs(output_path+subdirectory)
                try:
                    with open(output_path+subdirectory+'/'+i[:-4],'w') as output_file:
                        graph_dict, node_info_dict, edge_info_dict = parse_dot(dot_graph)
                        start_node = list(node_info_dict.keys())[0]
                        output_file.write("start||")
                        dfs_traversal(graph_dict, start_node, node_info_dict, edge_info_dict, output_file)
                    print("Output written to:", i)
                except Exception as e:
                    # Handle the exception here
                    error_message = f"Error processing doc_result[{i}]: {e}"
                    # Log the error message to the file
                    logging.error(error_message)
                    # Optionally, print the error message to the console
                    print(error_message)