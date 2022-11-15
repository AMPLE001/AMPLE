import re
import json
from graphviz import Digraph

#snakecase -> camelcase
def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def my_tokenizer(code):
    ## Remove code comments
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat,'',code)
    ## Remove newlines & tabs
    code = re.sub('(\n)|(\\\\n)|(\\\\)|(\\t)|(/)|(\\r)','',code)
    ## Mix split (characters and words)
    splitter = '\"(.*?)\"| +|(;)|(->)|(&)|(\*)|(\()|(==)|(~)|(!=)|(<=)|(>=)|(!)|(\+\+)|(--)|(\))|(=)|(\+)|(\-)|(\[)|(\])|(<)|(>)|(\.)|({)'
    code = re.split(splitter,code)
    ## Remove None type
    code = list(filter(None, code))
    code = list(filter(str.strip, code))
    # snakecase -> camelcase and split camelcase
    code_1 = []
    for i in code:
        code_1 += convert(i).split('_')
    #filt
    code_2 = []
    for i in code_1:
        if i in ['{', '}', ';', ':']:
            continue
        code_2.append(i)
    return(code_2)


def check(index_map, graph, gInput):
    for edge in graph:
        s, eType, e = edge
        if (s < 0 or s >= len(index_map)) or (e < 0 or e >= len(index_map)):
            raise(Exception("Graph Generation Error"))
        if eType not in [1,2,3,4,5]:
            raise(Exception("Edge Type Generation Error"))
    if len(gInput) != len(index_map):
        raise(Exception("Nodes Feature Num Error"))
    for node in gInput:
        if len(node) != 100:
            raise(Exception("Node Feature Generation Error"))
    return True

def get_nodes_by_key(nodes, key):
    for node in nodes:
        if node['key'].strip() == key:
            return node
    return None
    

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