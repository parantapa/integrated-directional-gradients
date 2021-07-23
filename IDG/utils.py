import torch
from IDG.models.modified_xlnet import XLNetForSequenceClassification
from IDG.models.bert_model import BertForSequenceClassification, BertConfig
from graphviz import Graph
import networkx as nx
from transformers import BertTokenizer, XLNetTokenizer

def read_graph(file):
    '''
    :param: File ... name of the file where the graph is stored with each line representing an edge as
    a node pair (u, v)
    :returns a networkx Digraph object
    '''
    G = nx.DiGraph()
    with open(file) as fs:
        for line in fs:
            u, v = line.strip().split(",")
            G.add_edge(int(u), int(v))

    return G


def find_children(G):
    t_sorted = list(nx.topological_sort(G))
    explored = {}
    children = {}
    for u in t_sorted:
        explored[u] = 1
        children[u] = []
        for v in nx.neighbors(G, u):
            if v not in explored:
                children[u].append(v)

    return children


def find_coalitions_graph(G):
    if not nx.is_directed_acyclic_graph(G):
        raise Exception("The input graph needs to be a directed acyclic graph")

    nodes = list(G.nodes())
    if min(nodes) != 0 or max(nodes) != len(nodes)-1:
        raise Exception("The node indices should start from 0 and end at n-1 ")

    t_sorted = list(nx.topological_sort(G))
    coalitions = []
    for n in t_sorted:
        coalitions.append([n] + list(nx.descendants(G, n)))

    return coalitions


def select_color(val, dir_):
    color_neg = "0.835 1.000 "
    color_pos = "0.334 1.000 "

    color = color_neg if dir_ < 0 else color_pos
    c_var = 1 - val
    c_h = c_var if c_var > 0.25 else 0.25
    c_h = f"{c_h:.3f}"

    return color + c_h


def visualize_tree(coalitions, val_score, div_dir, tree, path, chr_, view=False):
    '''
    :param coalitions: list of phrases/coalitions
    :param val_score: Value function scores for the coalitions
    :param div_dir: Sign of the dividend
    :param tree: list of edges of the parse tree
    :param path: path where the image is to be stored
    '''
    # lowest level --> tokens and their corresponding score
    tok_index = {}
    for i, c in enumerate(coalitions):
        if len(c) == 1:
            tok_index[c[0]] = i

    nodes = []
    leaves = coalitions[0]
    #print(coalitions)
    #print(val_score)
    d = Graph(filename=path+'idg_expl_image',
              node_attr={'style': 'filled', 'width': '0.15', 'height': '0.15', 'ordering': 'out',
                         'nodesep': '0.001', 'ranksep': '0.001', 'overlap': 'prism',
                         'overlap_scaling': '0.01', 'ratio': '0.2', 'format': 'png'})

    with d.subgraph() as s: # words are in the same level
        s.attr(rank='same')
        for word in leaves:
            w_ind = tok_index[word]
            lab = ''.join(k for k in word if not k.isdigit())
            s.node(chr_+'_'+'w_' + str(w_ind), label=lab, style='filled,solid', fillcolor='floralwhite', color='black')

    with d.subgraph() as s: # 1-1 map between words and the corresponding attribution score
        s.attr(rank='same')
        for word in leaves:
            w_ind = tok_index[word]
            v_score = f"{val_score[w_ind]:.3f}"
            dir_ = div_dir[w_ind]
            clr = select_color(val_score[w_ind], dir_)
            s.node(chr_+'_'+'n_' + str(w_ind), label=v_score, color=clr, fontcolor='black')
            nodes.append(chr_+'_'+'n_' + str(w_ind))
            d.edge(chr_+'_'+'n_' + str(w_ind), chr_+'_'+'w_' + str(w_ind))

    max_score = 0
    root = ''
    for u, v in tree: # rest of the edges in the tree
        if u is not None:
            u_n = chr_+'_'+'n_' + str(u)
            v_n = chr_+'_'+'n_' + str(v)
            if val_score[u] > max_score:
                max_score = val_score[u]
                root = u_n
            if val_score[v] > max_score:
                max_score = val_score[v]
                root = v_n
            s_u = f"{val_score[u]:.3f}"
            s_v = f"{val_score[v]:.3f}"
            d_u = div_dir[u]
            d_v = div_dir[v]

            if u_n not in nodes:
                clr = select_color(val_score[u], d_u)
                d.node(u_n, label=s_u, color=clr, fontcolor='grey99')
                nodes.append(u_n)

            if v_n not in nodes:
                clr = select_color(val_score[v], d_v)
                d.node(v_n, label=s_v, color=clr, fontcolor='grey99')
                nodes.append(v_n)

            d.edge(u_n, v_n)

    if not view:
        return d, root

    d.format = 'pdf'
    d.render()


def prepare_input(sentence, tokenizer):
    """
    Tokenizes, truncates, and prepares the input for modeling.
    NOTE: Requires Transformers>=3.0.0
    Parameters
    ----------
    sentence: str
        Input sentence to obtain sentiment from.
    tokenizer: XLNetTokenizer
        Tokenizer for tokenizing input.
    Returns
    -------
    features: dict
        Keys
        ----
        input_ids: torch.tensor(1, num_ids), dtype=torch.int64
            Tokenized sequence text.
        token_type_ids: torch.tensor(1, num_ids), dtype=torch.int64
            Token type ids for the inputs.
        attention_mask: torch.tensor(1, num_ids), dtype=torch.int64
            Masking tensor for the inputs.
    """
    features = tokenizer([sentence], return_tensors='pt', truncation=True, max_length=512)
    return features


def load_model(name=None, model_path=None, device=None):
    if name == 'xlnet':
        return load_xlnet_model(model_path, device)
    elif name == 'bert':
        return load_bert_model(model_path, device)
    else:
        pass


def load_xlnet_model(model_path, device):
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")
    model_states = torch.load(model_path, map_location=device)
    model.load_state_dict(model_states)
    model.eval()
    return model


def load_bert_model(model_path, device):
    config = BertConfig(vocab_size=30522, type_vocab_size=2)
    model = BertForSequenceClassification(config, 2, [11])
    model_states = torch.load(model_path, map_location=device)
    model.load_state_dict(model_states)
    model.eval()
    return model


def sequence_forward_func(inputs, model, tok_type_ids, att_mask):
    """
    Passes forward the inputs and relevant keyword arguments.
    Parameters
    ----------
    inputs: torch.tensor(1, num_ids), dtype=torch.int64
        Encoded form of the input sentence.
    tok_type_ids: torch.tensor(1, num_ids), dtype=torch.int64
        Tensor to specify token type for the model.
        Because sentiment analysis uses only one input, this is just a tensor of zeros.
    att_mask: torch.tensor(1, num_ids), dtype=torch.int64
        Tensor to specify attention masking for the model.

    Returns
    -------
    outputs: torch.tensor(1, 2), dtype=torch.float32
        Output classifications for the model.
    """
    outputs = model(inputs, token_type_ids=tok_type_ids, attention_mask=att_mask)
    if type(outputs) == tuple:
        return outputs[0]
    return outputs


def load_tokenizer(name=None):
    if name == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    elif name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    else:
        tokenizer = None
    return tokenizer


if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    #G.add_edge(2,1)
    print(find_coalitions_graph(G))