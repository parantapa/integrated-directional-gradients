from nltk.tree import Tree
from graphviz import Graph
import torch
from IDG.utils import sequence_forward_func, prepare_input, visualize_tree
from IDG.Intermediate_gradients.layer_intermediate_gradients import LayerIntermediateGradients
import torch.nn.functional as F
import IDG.parser as p_t


def calculate_IDG(inp, grads, position=None, bert=False):
    '''
    :param inp: Input at the embedding layer -> 1 x tokens x 768
    :param grads: Gradients calculated at different step sizes -> steps x tokens x 768
    :param position: Positions for which the score is to be calculated
    None -> scores of all the individual tokens
    list ->list of indices in the coalition
    :return: IDG score
    '''
    if position is None:
        norm_inp = inp / torch.norm(inp, dim=1).reshape(-1, 1)
        intm_grads = torch.sum(grads * norm_inp.unsqueeze(0), dim=2)
        div_features = torch.abs(torch.mean(grads, dim=0))
        div_score = torch.mean(intm_grads, dim=0)
        val_score = torch.abs(div_score) + torch.sum(div_features, dim=1)
        return div_score.detach().numpy().tolist(), val_score.detach().numpy().tolist(), \
               div_features.detach().numpy().tolist()


    else:
        if position == 'all':
            mask = torch.ones(inp.shape[0])
            if bert:
                mask[0] = 0
                mask[-1] = 0
            else:
                mask[-1] = 0 ## masking out the special tokens
                mask[-2] = 0
        else:
            position = torch.LongTensor(position)
            mask = torch.zeros(inp.shape[0])
            mask[position] = 1
        #print(mask)
        masked_inp = inp * mask.unsqueeze(1)
        norm_inp = masked_inp / torch.norm(masked_inp)
        intm_grads = torch.sum(grads * norm_inp.unsqueeze(0), dim=2)
        return torch.sum(torch.mean(intm_grads, dim=0))


def IDG_all(inp, base, grads, all_coalitions):
    '''
    :param inp: input at embedding layer -> 1 x tokens x 768
    :param base: baseline at embedding layer -> 1 x tokens x 768
    :param grads: Gradients calculated at different step sizes -> steps x tokens x 768
    :param all_coalitions: list of all possible coalitions (length > 1), each coalition is a tuple (i, j)
    :return: IDG score across all coalitions
    '''
    all_scores = calculate_IDG(inp, base, grads)
    coalition_scores = []
    for coalition in all_coalitions:
        score = calculate_IDG(inp, grads, coalition)
        all_scores = torch.cat((all_scores, score.unsqueeze(0)))
        coalition_scores.append(score.item())

    return torch.sum(all_scores), coalition_scores


def execute_single(string, model, tokenizer, target, path, bert=False):
    '''
    :param: string -> the parse string
    :param: model -> the classification model
    :param: tokenizer -> tokenier used by model
    :param: target -> the target class (0 -> negative, 1 -> positive)
    :param: path -> path where the image and the dot file will be saved
    '''
    n_steps = 250
    sequence = ' '.join(Tree.fromstring(string).leaves())
    features = prepare_input(sequence, tokenizer)
    input_ids = features["input_ids"]
    #print(f'input ids - {input_ids.shape}')
    token_type_ids = features["token_type_ids"]
    attention_mask = features["attention_mask"]

    baseline_ids = torch.zeros(input_ids.shape, dtype=torch.int64)
    # print(sequence_forward_func_loss(input_ids, model, token_type_ids, attention_mask, 0))
    dist = sequence_forward_func(input_ids, model, token_type_ids, attention_mask)
    dist = F.softmax(dist, dim=1)
    inferred_class = torch.argmax(dist).item()
    if inferred_class!=target:
        print('wrong inference')
    print(f'class distribution: {dist}')
    # instance of layer intermediate gradients based upon the dummy layer representing the embeddings
    #lig = LayerIntermediateGradients(sequence_forward_func, model.transformer.batch_first)
    if not bert:
        lig = LayerIntermediateGradients(sequence_forward_func, model.transformer.batch_first)
    else:
        lig = LayerIntermediateGradients(sequence_forward_func, model.bert.embeddings)
    grads_neg, step_sizes_neg, interm_neg, input_forward, baseline_forward = lig.attribute(inputs=input_ids,
                                                                                           baselines=baseline_ids,
                                                                                           target=inferred_class,
                                                                                           additional_forward_args=(
                                                                                               model, token_type_ids,
                                                                                               attention_mask),
                                                                                           n_steps=n_steps)


    inp = input_forward[0].squeeze(0) - baseline_forward[0].squeeze(0)
    p_tree, coalitions, tokenized_coalitions, indexed_coalitions = p_t.find_all_coalitions_indexed(string, tokenizer)
    dividend_func = [0 for _ in range(len(coalitions))]
    value_func = [0 for _ in range(len(coalitions))]
    dividend_dir = [0 for _ in range(len(coalitions))]
    div_token, val_token, div_features = calculate_IDG(inp, grads_neg, bert=bert)

    if not bert:
        z = sum([abs(i) for i in div_token[:-2]]) + sum([sum(lst) for lst in div_features[:-2]]) # neglecting the scores of the special tokens
    else:
        z = sum([abs(i) for i in div_token[1:-1]]) + sum([sum(lst) for lst in div_features[1:-1]])

    if bert:
        indexed_coalitions = [[ind+1 for ind in index] for index in indexed_coalitions]

    for i, index in enumerate(indexed_coalitions):
        if len(index) == 1:
            score = div_token[index[0]]
            dividend_func[i] = abs(div_token[index[0]])
        else:
            score = calculate_IDG(inp, grads_neg, index, bert=bert).item()
            dividend_func[i] = abs(score)
            z += abs(score)
        if score > 0:
            dividend_dir[i] = 1
        else:
            dividend_dir[i] = -1

    #z = sum(dividend_func)
    dividend_func = [s / z for s in dividend_func]
    div_token = [s/z for s in div_token]
    val_token = [s/z for s in val_token]

    children, root = p_t.find_children(p_tree)
    for i, c in enumerate(coalitions):
       if len(c) == 1:
           if len(indexed_coalitions[i]) == 1:
               value_func[i] = val_token[indexed_coalitions[i][0]]
           else:
               value_func[i] = dividend_func[i] + sum([val_token[j] for j in indexed_coalitions[i]])

    def calculate_value_function(n):
        if len(children[n]) == 0:
            return value_func[n]
        else:
            value_func[n] = dividend_func[n] + \
                            sum([calculate_value_function(children[n][i]) for i in range(len(children[n]))])
            return value_func[n]

    q = calculate_value_function(root)

    # Dump dividend scores from the embedding layer
    #with open(path+'out_pickle_file.p', 'wb') as ft:
    #    pickle.dump({'feature_dividend': div_features, 'true_cls': target, 'inferred_cls': inferred_class, 'z':z}, ft)

    visualize_tree(coalitions, value_func, dividend_dir, p_tree, path, 'a', view=True)


def execute_IDG(trees, model, tokenizer, target, path, bert=False):
    n_steps = 250
    all_tokens = []
    sentences = []
    if len(trees) == 1:
        execute_single(trees[0], model, tokenizer, target, path, bert=bert)
        return
    for tree in trees:
        tokens = Tree.fromstring(tree).leaves()
        all_tokens.extend(tokens)
        sentences.append(' '.join(tokens))
    sequence = ' '.join(all_tokens)
    #print(sequence)
    #print(sentences)

    tokenized_sents = [tokenizer.encode(sent, add_special_tokens=False) for sent in sentences]
    features = prepare_input(sequence, tokenizer)
    input_ids = features["input_ids"]
    token_type_ids = features["token_type_ids"]
    attention_mask = features["attention_mask"]
    baseline_ids = torch.zeros(input_ids.shape, dtype=torch.int64)
    # print(sequence_forward_func_loss(input_ids, model, token_type_ids, attention_mask, 0))
    dist = sequence_forward_func(input_ids, model, token_type_ids, attention_mask)
    dist = F.softmax(dist, dim=1)
    print(f'class distribution: {dist}')
    inferred_class = torch.argmax(dist).item()
    if inferred_class!=target:
        print('wrong inference')
    # instance of layer intermediate gradients based upon the dummy layer representing the embeddings
    if not bert:
        lig = LayerIntermediateGradients(sequence_forward_func, model.transformer.batch_first)
    else:
        lig = LayerIntermediateGradients(sequence_forward_func, model.bert.embeddings)
    grads_neg, step_sizes_neg, interm_neg, input_forward, baseline_forward = lig.attribute(inputs=input_ids,
                                                                                           baselines=baseline_ids,
                                                                                           target=inferred_class,
                                                                                           additional_forward_args=(
                                                                                               model, token_type_ids,
                                                                                               attention_mask),
                                                                                           n_steps=n_steps)

    inp = input_forward[0].squeeze(0) - baseline_forward[0].squeeze(0)
    dividend_func_all = []
    value_func_all = []
    dividend_dir_all = []
    sent_lengths = [len(sent) for sent in tokenized_sents]
    #print(f"sentence lengths - {sent_lengths}")
    div_token, val_token, div_features = calculate_IDG(inp, grads_neg, bert=bert)
    #z = sum([abs(i) for i in div_token[:-2]]) + sum([sum(lst) for lst in div_features[:-2]])
    if not bert:
        z = sum([abs(i) for i in div_token[:-2]]) + sum([sum(lst) for lst in div_features[:-2]]) # neglecting the scores of the special tokens
    else:
        z = sum([abs(i) for i in div_token[1:-1]]) + sum([sum(lst) for lst in div_features[1:-1]])

    all_coalitions = []
    p_trees = []
    ext = 0
    for t, tree in enumerate(trees):
        p_tree, coalitions, tokenized_coalitions, indexed_coalitions = p_t.find_all_coalitions_indexed(tree,
                                                                                                       tokenizer)
        p_trees.append(p_tree)
        all_coalitions.append(coalitions)
        if bert:
            indexed_coalitions = [[ind + 1 for ind in index] for index in indexed_coalitions]
        dividend_func = [0 for _ in range(len(coalitions))]
        value_func = [0 for _ in range(len(coalitions))]
        dividend_dir = [0 for _ in range(len(coalitions))]

        if t > 0:
            ext += sent_lengths[t-1]
            indexed_coalitions = [[ind + ext for ind in indices] for indices in indexed_coalitions]

        for i, index in enumerate(indexed_coalitions):
            #print(i, index)
            if len(index) == 1:
                score = div_token[index[0]]
                dividend_func[i] = abs(div_token[index[0]])
            else:
                score = calculate_IDG(inp, grads_neg, index, bert=bert).item()
                dividend_func[i] = abs(score)
                z += abs(score)
            if score > 0:
                dividend_dir[i] = 1
            else:
                dividend_dir[i] = -1
        dividend_func_all.append(dividend_func)
        dividend_dir_all.append(dividend_dir)

        children, root = p_t.find_children(p_tree)
        for i, c in enumerate(coalitions):
            if len(c) == 1:
                if len(indexed_coalitions[i]) == 1:
                    value_func[i] = val_token[indexed_coalitions[i][0]]
                else:
                    value_func[i] = dividend_func[i] + sum([val_token[j] for j in indexed_coalitions[i]])

        def calculate_value_function(n):
            if len(children[n]) == 0:
                return value_func[n]
            else:
                value_func[n] = dividend_func[n] + \
                                sum([calculate_value_function(children[n][i]) for i in range(len(children[n]))])
                return value_func[n]

        q = calculate_value_function(root)
        value_func_all.append(value_func)
    # calculate score for the whole text...<considering all the sentences>
    div_score = calculate_IDG(inp, grads_neg, position='all', bert=bert)
    if div_score.item() > 0:
        dividend_dir_all.append([1])
        clr = '0.334 1.000 0.250' # color of the root node for visualisation
    else:
        dividend_dir_all.append([-1])
        clr = '0.334 1.000 0.250'
    div_score = abs(div_score.item())
    z += div_score
    val_score = sum([v[0] for v in value_func_all])
    dividend_func_all.append([div_score])
    value_func_all.append([val_score])

    dividend_func_all = [[d/z for d in dividend] for dividend in dividend_func_all]
    value_func_all = [[v/z for v in value] for value in value_func_all]
    #print(value_func_all[-1])
    label = f'{value_func_all[-1][0]:.3f}' # label of the root
    viz_trees = []
    for i in range(len(tokenized_sents)):
        chr_ = chr(97 + i)
        t, node = visualize_tree(all_coalitions[i], value_func_all[i], dividend_dir_all[i],
                                 p_trees[i], 'dummy', chr_)
        viz_trees.append((t, node))

    t_d = Graph(filename=path+'idg_expl_image', node_attr={'style': 'filled', 'width': '0.15',
                                                        'height': '0.15'})

    t_d.node('r', label=label, color=clr, fontcolor='white')
    for t, n in viz_trees:
        t_d.subgraph(t)
        t_d.edge('r', n)

    t_d.format = 'pdf'
    t_d.render()
    # uncomment this part if you need the dividend scores from the embedding layer
    #with open(path+'.p', 'wb') as ft:
    #    pickle.dump({'feature_dividend': div_features, 'norm_factor': z, 'true_cls': target, 'inferred_cls': inferred_class}, ft)


def execute_IDG_from_Graph(G, inp, base, grads):
    '''
    parameters
    :param G: networkx Digraph object
    :grads: integrated gradients calculated for each node in the graph as numpy array
    '''
    raise NotImplementedError
