# coding=UTF-8
import argparse
import csv
from enum import unique
from genericpath import exists
from operator import index
#from numpy.core.fromnumeric import nonzero
import json
import numpy as np
import os
from gensim.models import Word2Vec
from tqdm import tqdm
import copy
import re
import nltk  
import warnings
from graphviz import Digraph
import pandas as pd
from utils import my_tokenizer, check


node_type_map = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5, 'SizeofOperand': 6,
    'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpression': 9, 'ParameterList': 10,
    'IdentifierDeclType': 11, 'SizeofExpression': 12, 'SwitchStatement': 13, 'IncDec': 14, 'Function': 15,
    'BitAndExpression': 16, 'UnaryExpression': 17, 'DoStatement': 18, 'GotoStatement': 19, 'Callee': 20,
    'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23, 'CFGErrorNode': 24, 'WhileStatement': 25,
    'InfiniteForNode': 26, 'RelationalExpression': 27, 'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30,
    'CompoundStatement': 31, 'UnaryOperator': 32, 'CallExpression': 33, 'CastExpression': 34,
    'ConditionalExpression': 35, 'ArrayIndexing': 36, 'PostIncDecOperationExpression': 37, 'Label': 38,
    'ArgumentList': 39, 'EqualityExpression': 40, 'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44,
    'ParameterType': 45, 'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49,
    'CastTarget': 50, 'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'File': 58, 'UnaryOperationExpression': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67, 'InitializerList': 68,
    'ElseStatement': 69
}

type_one_hot = np.eye(len(node_type_map))  

edgeType_full = {
    'IS_AST_PARENT': 1,
    'IS_CLASS_OF': 2,
    'FLOWS_TO': 3,
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
    'CONTROLS': 7,
    'DECLARES': 8,
    'DOM': 9,
    'POST_DOM': 10,
    'IS_FUNCTION_OF_AST': 11,
    'IS_FUNCTION_OF_CFG': 12
}

ver_edge_type = {
    '1' : 'IS_AST_PARENT',
    '2' : 'IS_CLASS_OF',
    '3' : 'FLOWS_TO',
    '4' : 'EDF',
    '5' : 'USE',
    '6' : 'REACHES',
    '7' : 'CONTROLS',
    '8' : 'DECLARES',
    '9' : 'DOM',
    '10' : 'POST_DOM',
    '11' : 'IS_FUNCTION_OF_AST',
    '12' : 'IS_FUNCTION_OF_CFG'
}

edgeType_reduced = {
    'IS_AST_PARENT': 1,
    'FLOWS_TO': 2,
    'REACHES': 3,
    'NSC': 4,
    'Self_loop': 5
}

ver_edge_type_reduced = {
    '1' : 'IS_AST_PARENT',
    '2' : 'FLOWS_TO',
    '3' : 'REACHES',
    '4' : 'NSC',
    '5' : 'Self_loop'
}
allowed_edge_types_reduced = {
    'IS_AST_PARENT': 'black',
    'FLOWS_TO': 'blue',
    'REACHES': 'red',
    'NSC':'red'
}

edge_type_ast = {
    'IS_AST_PARENT' : 'black',
    'FLOWS_TO': 'blue',
    'REACHES':'red',
    'NSC': 'green'
}

allowed_edge_types = {
    'FLOWS_TO': 'blue',  #3  red
    'REACHES': 'black',  #6  blue
    'CONTROLS': 'black',  #7   red
    'DOM': 'red',  #9  green
    'POST_DOM': 'red',  #10  green 
}

allowed_edge_types_full = {
    'IS_AST_PARENT': 'black',
    'IS_CLASS_OF': 'purple',
    'FLOWS_TO': 'blue',
    'DEF': 'green',
    'USE': 'green',
    'REACHES': 'black',
    'CONTROLS': 'black',
    'DECLARES': 'green',  #
    'DOM': 'red',
    'POST_DOM': 'red',
    'IS_FUNCTION_OF_AST': 'green',
    'IS_FUNCTION_OF_CFG': 'green'
}

allowed_edge_types_dom = {
    'DOM' : 'black',
    'POST_DOM' : 'red'
}

#data dependency
allowed_edge_types_reach = {  
    'REACHES' : 'black'
}

allowed_edge_types_control = {
    'CONTROLS' : 'black'
}

allowed_edge_types_def_use = {
    'DEF' : 'black',
    #'USE' : 'red'
}

allowed_edge_types_ast = {
    'IS_AST_PARENT' : 'black',
    #'USE' : 'red'
}

def checkVul(cFile):
    with open(cFile, 'r') as f:
        fileString = f.read()
        return (1 if "BUFWRITE_COND_UNSAFE" in fileString or "BUFWRITE_TAUT_UNSAFE" in fileString else 0)
warnings.filterwarnings('ignore')

def read_csv(csv_file_path):
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data

def build_ast(starts, edges, ast_edges):
    if len(starts) == 0:
        return 
    new_starts = []
    for i in starts:
        ast = {}
        ast['start'] = i
        ast['end'] = []
        for edge in edges:
            if edge['start'].strip() == i and edge['type'].strip() == 'IS_AST_PARENT':
                ast['end'].append(edge['end'].strip())
                new_starts.append(edge['end'].strip())
        if len(ast['end']) > 0:
            ast_edges.append(ast)
    build_ast(new_starts, edges, ast_edges)
    pass

def get_nodes_by_key(nodes, key):
    for node in nodes:
        if node['key'].strip() == key:
            return node
    return None

def check_def(nodes, edges):
    defed = []
    for e in edges:
        if e['type'] == 'DEF':
            defed.append(e['end'])

def combine(x_node, y_node):
    x_type = x_node['type'].strip()
    y_type = y_node['type'].strip()
    if x_type == 'ExpressionStatement':
        if y_type == 'AssignmentExpression': 
            return True
        if y_type == 'UnaryExpression' and x_node['code'].strip() == y_node['code'].strip():  
            return True
        if y_type == 'PostIncDecOperationExpression' and x_node['code'].strip() == y_node['code'].strip():  
            return True
        if y_type == 'CallExpression':   
            return True
    
    if x_type == 'IdentifierDeclStatement' and y_type == 'IdentifierDecl': 
        return True
    
    if x_type == 'CallExpression' and y_type == 'ArgumentList':  
        return True
    if x_type == 'Callee' and y_type == 'Identifier':  
        return True
    if x_type == 'Argument' and x_node['code'].strip() == y_node['code'].strip():  
        return True
    if x_type == 'Condition' and x_node['code'].strip() == y_node['code'].strip():  
        return True
    if x_type == 'ForInit':  
        if x_node['code'].strip() == y_node['code'].strip(): 
            return True
        if y_type == 'AssignmentExpression':
            return True
    
    return False

def check_dup_node(duplist, node_key):
    for sublist in duplist:
        head = sublist[0]
        if node_key in sublist:
            return head
    return None        

def ast_prune(nodes, ast_edges, ast_id):
    ast_new_edges = []
    duplist = []
    first_flag = True
    for item in ast_edges:
        repeat = []
        x = item['start']
        x_node = get_nodes_by_key(nodes, x)
        x_node_key = x_node['key'].strip()
        repeat.append(x_node_key)
        edges = {}  
        p_node = x_node_key  
        t_node = copy.deepcopy(item['end'])
        for y in item['end']:
            y_node = get_nodes_by_key(nodes, y)
            y_node_key = y_node['key'].strip()
            if combine(x_node, y_node) == True:   
                t_node.remove(y_node_key)
                flag = False
                for sublist in duplist:  
                    if x_node_key in sublist:
                        sublist.append(y_node_key)
                        p_node = sublist[0]  
                        flag = True
                        break
                if not flag:  
                    repeat.append(y_node_key)
            else:  
                head = check_dup_node(duplist, x_node_key)
                if head != None:
                    p_node = head
        if len(repeat) > 1:
            duplist.append(repeat)
        edges = {}
        edges['start'] = p_node
        edges['end'] = t_node

        if first_flag == True:
            first_flag = False
            edges['start'] = x_node_key
            ast_new_edges.append(edges)
        elif p_node == x_node_key and len(t_node) > 0: 
            ast_new_edges.append(edges)
        elif p_node != x_node_key:  
            find = False
            for i in ast_new_edges:   
                if i['start'] == p_node:  
                    find = True
                    i['end'] += t_node
                    break
            if find == False:
                ast_new_edges.append(edges)
    return ast_new_edges

def edge_ver(index_map, edges):
    new_edges = []
    for e in edges:
        start, eType, end = e
        start = index_map[start]
        end = index_map[end]
        new_edge = [start, eType, end]
        new_edges.append(new_edge)
    return new_edges

def count_nodes_edges(edges):
    nodes = set()
    edges_num = 0
    for item in edges:
        nodes.add(item['start'])
        edges_num += len(item['end'])
        for y in item['end']:
            nodes.add(y)
    return len(nodes), edges_num  

def get_ncs_edges(all_ast_edges, nsc_type, nodes):
    nsc_edges = list()
    first = True  #flag
    #sent_order_dup = sorted(set(sent_order), key = sent_order.index)
    par_sent = []
    tmp_sent = copy.deepcopy(all_ast_edges)
    for ast_sent in all_ast_edges:
        if get_nodes_by_key(nodes,ast_sent[0]['start'])['type'] == 'Parameter':
            par_sent.append(ast_sent)
            tmp_sent.remove(ast_sent)
    all_ast_edges = par_sent + tmp_sent

    for ast_sent in all_ast_edges:
        #判断一个句子内
        s_nodes =[]
        t_nodes = []
        for ast in ast_sent:
            s_node = ast['start']
            s_nodes.append(s_node)
            t_nodes = t_nodes + ast['end']
        nsc_nodes = [] 
        if s_nodes == t_nodes and len(s_nodes) == 1:  #break; continue; return; 
            nsc_nodes = t_nodes
        if len(s_nodes) == 1 and len(t_nodes) == 0:  # if(a) {start:'a', end:[]}
            nsc_nodes = s_nodes
        else:  
            for node in t_nodes:
                if node not in s_nodes:
                    nsc_nodes.append(int(node))
        nsc_nodes.sort(reverse=False)
        idx = 0
        if first:
            first = False
            idx = 1
            s = nsc_nodes[0]  
        for i in range(idx, len(nsc_nodes)):
            #if int(s) >= int(nsc_nodes[i]):
            #    ex = Exception("Sequence Order Error!!")
            #    raise(ex)
            edge = [str(s), nsc_type, str(nsc_nodes[i])]
            nsc_edges.append(edge)
            s = nsc_nodes[i]
    return nsc_edges

def get_combine_ncs_edges(all_ast_edges, nsc_type, nodes, same_var):
    nsc_edges = list()
    first = True  #flag
    #sent_order_dup = sorted(set(sent_order), key = sent_order.index)
    par_sent = []
    tmp_sent = copy.deepcopy(all_ast_edges)
    for ast_sent in all_ast_edges:
        if get_nodes_by_key(nodes,ast_sent[0]['start'])['type'] == 'Parameter':
            par_sent.append(ast_sent)
            tmp_sent.remove(ast_sent)
    all_ast_edges = par_sent + tmp_sent
    record_nodes = set()
    for ast_sent in all_ast_edges:
        s_nodes =[]
        t_nodes = []
        for ast in ast_sent:
            s_node = ast['start']
            s_nodes.append(s_node)
            t_nodes = t_nodes + ast['end']
        nsc_nodes = []  
        if s_nodes == t_nodes and len(s_nodes) == 1:  #break; continue; return; 
            nsc_nodes = t_nodes
        if len(s_nodes) == 1 and len(t_nodes) == 0:  # if(a) {start:'a', end:[]}
            nsc_nodes = s_nodes
        else:  
            for node in t_nodes:
                if node not in s_nodes:
                    nsc_nodes.append(int(node))
        
        nsc_nodes.sort(reverse=False)
        nsc_nodes = [str(i) for i in nsc_nodes]
      
        combine_nsc_nodes = []
        for i, node in enumerate(nsc_nodes):
            tmp = check_dup_node(same_var, node)
            if tmp == None:
                raise(Exception("None Error!"))
            if tmp in record_nodes:
                continue
            record_nodes.add(tmp)
            combine_nsc_nodes.append(tmp)
        idx = 0
        if first:
            first = False
            idx = 1
            s = combine_nsc_nodes[0]  
        for i in range(idx, len(combine_nsc_nodes)):
            edge = [str(s), nsc_type, str(combine_nsc_nodes[i])]
            nsc_edges.append(edge)
            s = combine_nsc_nodes[i]
    return nsc_edges

def spe_sent(n_type, n_code):
    if n_type == 'BreakStatement' and n_code == ['break', ';']:
        return True
    elif n_type == 'ContinueStatement' and n_code == ['continue', ';']:
        return True
    elif n_type == 'ReturnStatement' and n_code == ['return', ';']:
        return True
    elif n_type == 'InfiniteForNode' and n_code == ['true']:  #(;;)
        return True
    elif n_type == 'Label' and 'case' in n_code: #case 1:
        return True
    return False

def get_same_var(all_edges, nodes):
    all_nodes = set()
    for ast in all_edges:
        for item in ast:
            all_nodes.add(item['start'])
            all_nodes = all_nodes | set(item['end'])
    same_var = []
    for node in all_nodes:
        flag = False
        x_node = get_nodes_by_key(nodes, node)
        for item in same_var:
            for y in item:
                y_node = get_nodes_by_key(nodes, y)
                if x_node['code'].strip() == y_node['code'].strip() and x_node['type'].strip() in ['Argument','Identifier'] and y_node['type'].strip() in ['Argument','Identifier']:
                    item.append(x_node['key'].strip()) 
                    flag = True
                    break
            if flag:
                break
        if not flag:
            same_var.append([x_node['key'].strip()])
    sorted_same_var = []
    for i in same_var:
        sorted_same_var.append(sorted(i))    
    return sorted_same_var

def var_combine(ast_edges, same_var):
    ast_combine_edges = copy.deepcopy(ast_edges)
    for ast_sent in ast_combine_edges:
        if len(ast_sent) == 1 and len(ast_sent[0]['end']) == 1 and ast_sent[0]['start'] == ast_sent[0]['end'][0]:
            continue
        for item in ast_sent:
            heads = check_dup_node(same_var, item['start'])
            if heads != None:
                item['start'] = heads
            for i, y_node in enumerate(item['end']):
                heade = check_dup_node(same_var, y_node)
                if heade != None:
                    item['end'][i] = heade
    return ast_combine_edges

def graphGeneration(nodes, edges, edge_type_map, ver_edge_type_map):
    index_map = dict()
    index_map_ver = dict()
    all_nodes = set()
    all_ast_edges = []
    all_edges = list()
    for node in nodes:
        if node['isCFGNode'].strip() != 'True' or node['key'].strip() == 'File':
            continue
        all_nodes.add(node['key'])  
        # entry  exit
        if node['type'] in ['CFGEntryNode','CFGExitNode']:
            continue
        nodeKey = [node['key']]
        ast_edges = []
        build_ast(nodeKey, edges, ast_edges)
  
        if len(ast_edges) == 0:
            #break; continue; returns; (;;) 
            if spe_sent(node['type'], node['code'].strip().split()):
                dic = {}
                dic['start'] = nodeKey[0]
                dic['end'] = nodeKey
                ast_edges.append(dic)
            else:
                return None, None, None, True
        all_ast_edges.append(ast_edges)  
    edge_count_before = 0
    node_count_before = set()
    for item in all_ast_edges:
        for ast in item:
            node_count_before.add(ast['start'])
            node_count_before = node_count_before | set(ast['end'])
            edge_count_before += len(ast['end'])
    step1_ast_edges = []
    for i in all_ast_edges:    
        # break; continue; return
        if len(i) == 1 and i[0]['start'] == i[0]['end'][0] and len(i[0]['end']) == 1:
            step1_ast_edges.append(i)
        else:
            new_edges = ast_prune(nodes, i, 'IS_AST_PARENT')
            step1_ast_edges.append(new_edges)
    same_var = get_same_var(step1_ast_edges, nodes)
    step2_ast_edges = var_combine(step1_ast_edges, same_var)

    edge_count_after = 0
    node_count_after = set()
    for item in step2_ast_edges:
        for ast in item:
            node_count_after.add(ast['start'])
            node_count_after = node_count_after | set(ast['end'])
            edge_count_after += len(ast['end'])
    #add nsc
    nsc_edges_ = get_ncs_edges(step1_ast_edges, 'NSC', nodes)
    nsc_edges = get_combine_ncs_edges(step1_ast_edges, 'NSC', nodes, same_var)
    ast_type = 'IS_AST_PARENT'

    for item in step2_ast_edges:
        # break; continue; return; 
        if len(item) == 1 and len(item[0]['end']) == 1 and item[0]['start'] == item[0]['end'][0]:
            continue
        for x in item:
            start = x['start']
            for end in x['end']:
                all_edges.append([start, ast_type, end])  

    for e in edges:
        start, end, eType = e['start'], e['end'], e['type']    
        start_node = get_nodes_by_key(nodes, start)
        end_node = get_nodes_by_key(nodes, end)
        if start_node['isCFGNode'].strip() != 'True' or end_node['isCFGNode'].strip() != 'True':
            continue
        if eType != 'IS_FILE_OF' and eType != ast_type:
            if not eType in edge_type_map: #or not start in all_nodes or not end in all_nodes:
                continue
            all_edges.append([start, eType, end])  
    for e in all_edges:
        start, _, end = e
        all_nodes.add(start)
        all_nodes.add(end)
    if len(all_nodes) == 0 or len(all_nodes) > 500:  # V1是750
        return None, None, None, None
    for i, node in enumerate(all_nodes):
        index_map[node] = i
        index_map_ver[i] = node
    all_edges_new = []   #original full graph
    for e in all_edges: # e = [start, type, end]
        e_new = [index_map[e[0]], edge_type_map[e[1]], index_map[e[2]]]
        all_edges_new.append(e_new)
    #add nsc_edge
    for e in nsc_edges:
        e_new = [index_map[e[0]], edge_type_map[e[1]], index_map[e[2]]]
        all_edges_new.append(e_new)
    #add self-loop
    loop = 'Self_loop'
    for node in all_nodes:
        self_loop = [index_map[node], edge_type_map[loop], index_map[node]]
        all_edges_new.append(self_loop)
    if len(all_edges_new) == 0:
        return None, None, None, None
    edges_num = {}
    for t in edge_type_map.keys():
        edges_num[t] = 0
    for e in all_edges_new:
        key = ver_edge_type_map[str(e[1])]
        edges_num[key] += 1
    edges_num['ast_reduced'] = edge_count_before - edge_count_after
    edges_num['nsc_reduced'] = len(nsc_edges_) - len(nsc_edges)
    edges_num['nodes_reduced'] = len(node_count_before) - len(node_count_after)
    return index_map_ver, all_edges_new, len(index_map_ver), edges_num

def word2vec(nodes, index_map, graph, wv):
    gInput = list()
    all_nodes = set()
    for item in graph:
        s, _, e = item
        all_nodes.add(e)
        all_nodes.add(s)
    if len(all_nodes) != len(index_map):
        print("Process Error!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None
    for i in index_map:
        true_id = index_map[i]
        node = get_nodes_by_key(nodes, true_id)
        node_content = node['code'].strip()
        #node_type = node_type_map[node['type'].strip()] - 1
        #one_hot = np.zeros(len(node_type_map))
        #one_hot[node_type] = 1.0
        #tokens = nltk.word_tokenize(node_content)
        tokens = my_tokenizer(node_content)
        nrp = np.zeros(100)
        for token in tokens:
            try:
                embedding = wv.wv[token]
            except:
                embedding = np.zeros(100)
            nrp = np.add(nrp, embedding)
        if len(tokens) > 0:
            fnrp = np.divide(nrp, len(tokens))
        else:
            fnrp = nrp
        #fnrp = np.concatenate((fnrp, one_hot), axis = 0)
        gInput.append(fnrp.tolist())
    return gInput

def visual_graph(all_nodes, index_map, edges, allowed_edge_types, ver_edge_type,  file_name = 'graph'):
    graph = Digraph()
    nodes = set()
    for edge in edges:
        start, t, end = edge
        nodes.add(start)
        nodes.add(end)
        type_ = ver_edge_type[str(t)]
        if type_ in allowed_edge_types.keys():
            color = allowed_edge_types[type_]
            s = index_map[start]
            e = index_map[end]
            graph.edge(s, e, color=color, label=type_)
    for node in nodes:
        true_id = index_map[node]
        #node_content = all_nodes[true_id]
        node_content = get_nodes_by_key(all_nodes, true_id)
        graph.node(name = true_id, label = true_id + '\n' + node_content['code'].strip() + '\n' + str(node_content['type'].strip()))
    graph.render(file_name, view = False)

def visual():
    #file_name = "qemu17498_0.c"
    file_name = "sample.c"
    file_path = "reveal_dataset/sample"
    nodes_path = os.path.join(file_path,  "nodes.csv")
    edges_path = os.path.join(file_path,  "edges.csv")
    edges = read_csv(edges_path)  #{'start': '6478', 'end': '6479', 'type': 'IS_AST_PARENT', 'var': ''}
    nodes = read_csv(nodes_path)  
    index_map, graph, nodes_num, reduced_num = graphGeneration(nodes, edges, edgeType_reduced, ver_edge_type_reduced)
    pdnodes = pd.DataFrame(nodes)
    pdnodes.to_csv("graph/new_nodes.csv")
    pdedges = pd.DataFrame(edges)
    pdedges.to_csv("graph/new_edges.csv")
    if index_map != None and graph != None: 
        visual_graph(nodes, index_map, graph, edge_type_ast, ver_edge_type_reduced, "graph/final_reduced_"+file_name)
        #visual_graph_full(nodes_ids_to_nodes, edges, allowed_edge_types_reduced, "graph_reduced")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='normalized csv files to process', default='../devign_dataset/devign_parsed/')  #nodes  edges
    parser.add_argument('--src', help='source c files to process', default='../devign_dataset/devign_raw_code/')
    parser.add_argument('--json_files', help = 'train and test and valid', default=['devign_dataset/devign_data_split/train_raw_code.json', 'devign_dataset/devign_data_split/test_raw_code.json', 'devign_dataset/devign_data_split/valid_raw_code.json'])
    parser.add_argument('--wv', default='devign_dataset/devign_wv_models/devign_train_subtoken_data')  # word2vec
    parser.add_argument('--output_dir', default='devign_dataset/devign_cpg_c2_2')
    args = parser.parse_args()
    model = Word2Vec.load(args.wv)
    train_path, test_path, valid_path = args.json_files  #[0], args.json_files[1], args.json_files[2]
    train_data = []
    test_data = []
    valid_data = []
    with open(train_path, 'r') as f:
        train_data = json.load(f)
        #for line in f.readlines():
        #    train_data.append(json.loads(line))
    with open(test_path, 'r') as f:
        test_data = json.load(f)
        #for line in f.readlines():
        #    test_data.append(json.loads(line))
    with open(valid_path, 'r') as f:
        valid_data = json.load(f)
        #for line in f.readlines():
        #    valid_data.append(json.loads(line)) 
    data = [train_data, test_data, valid_data]  #all data
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print(train_data[0])
    print("*"*100)
    train_output_path = open(os.path.join(output_dir, 'devign-train-v2.json'), 'w')
    test_output_path = open(os.path.join(output_dir, 'devign-test-v2.json'), 'w')
    valid_output_path = open(os.path.join(output_dir, 'devign-valid-v2.json'), 'w')
    output_files = [train_output_path, test_output_path, valid_output_path]

    train_num_path = open(os.path.join(output_dir, 'devign-train-num-v2.json'), 'w')
    test_num_path = open(os.path.join(output_dir, 'devign-test-num-v2.json'), 'w')
    valid_num_path = open(os.path.join(output_dir, 'devign-valid-num-v2.json'), 'w')
    num_files = [train_num_path, test_num_path, valid_num_path]

    train_file = open(os.path.join(output_dir, 'devign-train-file.json'), 'w')
    test_file = open(os.path.join(output_dir, 'devign-test-file.json'), 'w')
    valid_file = open(os.path.join(output_dir, 'devign-valid-file.json'), 'w')
    file_names = [train_file, test_file, valid_file]

    bad_file = []
    bad_file_path = open(os.path.join(output_dir, 'bad_file.json'), 'w')
    for i in range(len(data)):
        print("!!!")
        final_data = []
        final_num = []
        files = []
        num = 0
        for _, entry in enumerate(tqdm(data[i])):
            
            file_name = entry['file_path']
            nodes_path = os.path.join(args.csv, file_name, 'nodes.csv')
            edges_path = os.path.join(args.csv, file_name, 'edges.csv')
            label = int(entry['label'])
            if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
                continue
            nodes = read_csv(nodes_path)
            edges = read_csv(edges_path)
            index_map, graph, nodes_num, edges_num = graphGeneration(nodes, edges, edgeType_reduced, ver_edge_type_reduced)
            if edges_num == True:
                bad_file.append(file_name)
            if index_map is None or graph is None or nodes_num is None or edges_num is None:
                continue
            gInput = word2vec(nodes, index_map, graph, model)
            if gInput is None:
                continue
            if check(index_map, graph, gInput) != True:
                print("check error!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                continue
            data_point = {
                'node_features': gInput,
                'graph': graph,
                'targets': [[label]]
            }
            num_point = {
                'file_name' : file_name,
                'nodes_num' : nodes_num,
                'edges_num' : edges_num
            }
            num += 1
            files.append(file_name)
            final_data.append(data_point)
            final_num.append(num_point)
        print(num)
        json.dump(final_data, output_files[i])
        json.dump(final_num, num_files[i])
        json.dump(files, file_names[i])
        output_files[i].close()
        num_files[i].close()
        file_names[i].close()
    json.dump(bad_file, bad_file_path)
    bad_file_path.close()

if __name__ == '__main__':
    main()
    #visual()