import re
from nltk.tree import Tree, ParentedTree


def is_contained(lst1, lst2):
    # check whether lst2 is completely contained in lst1
    l = [x for x in lst2 if x in lst1]
    return len(l) == len(lst2)


def find_all_combinations(string, tag=False):
    '''
    :param string: parse tree which can be directly read by tree object
    :param tag: if the string has POS tags
    :return: all the valid sub phrases as list of list of strings
    '''
    tree = Tree.fromstring(string)
    counter = 1
    all_nodes = []
    all_nodes.append((tree.leaves(), None, 0))
    trees = [(tree, 0)]
    root = True
    while len(trees) > 0:
        tree, parent = trees.pop(0)
        for subtree in tree:
            all_nodes.append((subtree.leaves(), parent, counter))
            if len(subtree.leaves()) > 1:
                trees.append((subtree, counter))
            counter += 1
        # print(all_nodes)
    parse_tree = [(x[1], x[2]) for x in all_nodes]
    coalitions = [x[0] for x in all_nodes]
    return parse_tree, coalitions

    #levels = []
    #for i in range(tr.height(), 0, -1):
    #    for t in tr.subtrees(lambda x: x.height() == i):
    #        levels.append(t.leaves())

    #if not tag:
    #    return levels
    #else:
    #    levels.sort()
    #    levels = list(levels for levels, _ in itertools.groupby(levels))
    #    return sorted(levels, key=lambda x: len(x), reverse=True)


def check_valid_phrase():
    '''
    :param phrase_index: list of tokens for the phrase
    :return:
    '''
    raise NotImplementedError


def find_sub_coalitions(phrase, coalitions):
    '''
    :param phrase: the phrase for which the sub-coalitions are to be calculated
    it could be int which represents the index of the phrase in the coalitions set
    or a string representing the phrase itself.
    :param tokenized_coalitions: all the coalitions tokenized by tokenizer
    :param tokenizer to use for encoding
    :return: all the subcoalitions for the given phrase, list of indices which can then
    be directly used for masking
    '''
    if type(phrase) == int:
        phrase_ = coalitions[phrase]
    else:
        phrase_ = phrase.strip().split()
    sub_coalitions = []
    if phrase_ in coalitions:
        for i, c in enumerate(coalitions):
            if len(c)<=len(phrase_):
                if is_contained(phrase_, c):
                    sub_coalitions.append(i)
    return sub_coalitions


def encode_position_parse_tree(string):
    '''
    :param string: parse tree string that can be directly read as nltk tree object
    :return: Parse tree string with positions encoded. e.g., if word 'with' occurs twice then replace the second
    occurrence with 'with1'
    '''
    matches = re.finditer("[^0-9][a-zA-Z!./*&,:;-]+", string)
    words = []
    for match in matches:
        words.append(match.group())
    m_string = re.sub("[^0-9][a-zA-Z!./*&,:;-]+", "%s", string)
    word_dict = {}
    words_p = []
    for word in words:
        if word not in word_dict:
            words_p.append(word)
            word_dict[word] = 1
        else:
            words_p.append(word + str(word_dict[word]))
            word_dict[word] += 1
    return m_string % tuple(words_p)


def word_token_map(string, tokenizer):
    words = Tree.fromstring(string).leaves()
    sentence = ' '.join(words)
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    w_c = 0
    i = 0
    word_token_map = {}
    unique_word = {}
    curr_token = []
    curr_index = []
    while i < len(tokens):
        curr_token.append(tokens[i])
        curr_index.append(i)
        word = tokenizer.decode(curr_token)
        if word == words[w_c] or word.replace(' ','') == words[w_c].lower():
            word = words[w_c]
            w_c += 1
            if word not in unique_word:
                word_token_map[word] = (curr_token, curr_index)
                unique_word[word] = 1
            else:
                word_token_map[word + str(unique_word[word])] = (curr_token, curr_index)
                unique_word[word] += 1
            curr_token = []
            curr_index = []
        i += 1
    return word_token_map


def find_all_coalitions_indexed(string, tokenizer):
    '''
    returns all the possible coalitions with tokens replaced by their corresponding index in the
    input
    '''
    p_string = encode_position_parse_tree(string)
    parse_tree, coalitions = find_all_combinations(p_string, tag=True)
    w_t_map = word_token_map(string, tokenizer)
    tokenized_coalitions = []
    indexed_coalitions = []
    for c in coalitions:
        coalition_tok = []
        coalition_ind = []
        for w in c:
            coalition_tok.extend(w_t_map[w][0])
            coalition_ind.extend(w_t_map[w][1])
        tokenized_coalitions.append(coalition_tok)
        indexed_coalitions.append(coalition_ind)
    return parse_tree, coalitions, tokenized_coalitions, indexed_coalitions


def find_children(p_tree):
    children = {}
    for u, v in p_tree:
        if u is None:
            root = v
            continue
        else:
            if u not in children:
                children[u] = [v]
            else:
                children[u].append(v)
            if v not in children:
                children[v] = []

    return children, root
