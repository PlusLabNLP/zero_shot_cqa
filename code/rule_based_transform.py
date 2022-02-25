import torch
import sys
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import sys

wh_pos = ['WDT', 'WP',  'WP$', 'WRB']
ques_sent = ['SINV', 'SQ', 'SBARQ']

def find_pp_sbar(children, once=False, wh_word=None):
    new_tokens = []
    for child in children:
        if child['word'] == '?' or child['word'] == '.':
            continue
        if (child['nodeType'] == 'SBAR' or child['nodeType'] ==  'PP') and child['word'] != 'for' and child['word'] != 'with':
            if once is False:
                once = True
                new_tokens.append(wh_word.replace('what', '[MASK]') + ' ' + child['word'])
            else:
                new_tokens.append(child['word'])
        elif 'children' in child:
            ctoks, once = find_pp_sbar(child['children'], once, wh_word)
            new_tokens.append(ctoks)
        else:
            new_tokens.append(child['word'])
    return ' '.join(new_tokens), once

def append_word(root):
    if not 'children' in root:
        return root['word'] if root['word'] != '?' else ''
    
    new_tokens = []
    for child in root['children']:
        ctoken = append_word(child)
        if ctoken != '':
            new_tokens.append(ctoken)
    return ' '.join(new_tokens)

def process_sq(child, wh_word, what_taken):
    new_tokens = []
    if len(child['children']) > 1:
        ctokens = (child['children'][1]['word'])
        if ctokens != '?':
            new_tokens.append(ctokens)
        word = child['children'][0]['word']
        if word == 'do' or word == 'does' or word == 'did': 
            pass
        else:
            ctokens = (child['children'][0]['word'])
            if ctokens != '?':
                new_tokens.append(ctokens)
        if 'what' in wh_word:
            new_tokens.append(find_pp_sbar(child['children'][2:], what_taken, wh_word)[0])
        else:
            for chd in child['children'][2:]:
                new_tokens.append(append_word(chd))
    else:
        new_tokens.append(child['word'])
    return ' '.join(new_tokens)

def is_np(root):
    if not 'children' in root:
        return False
    for child in root['children']:
        if child['nodeType'] == 'NP':
            return True
    return False
    
def tree_process_sq(root):
    if not 'children' in root:
        return root['word'], None
    new_tokens = []
    wh_word = None
    wh_wordcnt = 0
    for ccnt, child in enumerate(root['children']):
        if child['nodeType'] == 'SQ' and ccnt > 0:
            wh_word = (root['children'][ccnt-1]['word']).lower()
            wh_wordcnt = ccnt-1
            break

    if wh_word == 'who':
        wh_word = 'what'
    if wh_word == 'which':
        wh_word = 'what'

    cwh_words = None
    append_last = False
    what_taken = False
    for ccnt, child in enumerate(root['children']):
        if child['nodeType'] == 'SQ':
            new_tokens.append(process_sq(child, wh_word, what_taken))
        elif wh_word is not None and ccnt ==wh_wordcnt and (wh_word in ['why', 'how', 'where', 'when'] or 'how' in wh_word):
            append_last = True
        elif wh_word is not None and ccnt ==wh_wordcnt and 'what' in wh_word:
            if is_np(root['children'][ccnt+1]):
                append_last = True
            else:
                new_tokens.append(wh_word.replace('what', '[MASK]')) #tree_process_sq(child))
                what_taken = True
        else:
            ctokens, cwh_word = (tree_process_sq(child))
            if ctokens != '?':
                new_tokens.append(ctokens) #tree_process_sq(child))
            if cwh_word is not None:
                cwh_words = cwh_word
    wh_word = cwh_words if wh_word is None else wh_word
    new_tokens = ' '.join(new_tokens)
    if append_last:
        if wh_word == 'why':
            new_tokens += ' because [MASK]'
        elif 'how' in wh_word:
            if wh_word == 'how often':
                new_tokens += ' [MASK]'
            elif wh_word == 'how long':
                new_tokens += ' for [MASK]'
            else:
                new_tokens += ' by ' + wh_word.replace('how', '[MASK]')
            #how often how long
        elif wh_word =='where':
            new_tokens += ' at [MASK]'
        elif wh_word =='when':
            new_tokens += ' when [MASK]'
        elif 'what' in wh_word:
            if not '[MASK]' in new_tokens:
                new_tokens += ' ' + wh_word.replace('what', '[MASK]') #' when [MASK]'
    return new_tokens, wh_word

def process_nowh(tokens, pos_tags):
    new_tokens = []
    cnt = 0
    for t, pos in zip(tokens[::-1], pos_tags[::-1]):
        if not pos in wh_pos or cnt > 0:
            if t != '?':
                new_tokens.append(t)
        else:
            new_tokens.append('[MASK]')
            cnt += 1
    new_tokens = new_tokens[::-1]
    if cnt == 0:
        new_tokens.append('[MASK]')
    return ' '.join(new_tokens)

load_data = open(sys.argv[1]).readlines()
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
replace_dict = {'which of these': 'what', 'which of the following': 'what', 'Which of these': 'What', 'Which of the following': 'What'}
for data in load_data:
    for key in replace_dict:
        data=data.replace(key, replace_dict[key])
    a=predictor.predict(data)
    tokens, pos_tags, hierplane_tree, trees = a['tokens'], a['pos_tags'], a['hierplane_tree'], a['trees']
    new_tokens = []
    if hierplane_tree['root']['nodeType'] == 'SQ': #yes/no
        new_sent = ' '.join(tokens) + ' [MASK] .'
        print(new_sent)
        continue
    
    new_tokens, wh_word = tree_process_sq(hierplane_tree['root'])
    if wh_word is None:
        new_sent = process_nowh(tokens, pos_tags)
    else:
        new_sent = tree_process_sq(hierplane_tree['root'])[0]
    new_sent = ' '.join(new_sent.split())
    if new_sent.count('[MASK]')!=1:
        new_sent = ' '.join(tokens) + ' [MASK] .'
    if new_sent[-1] == '?':
        new_sent[-1] = '.'
    if new_sent[-1] != '.':
        new_sent = new_sent + ' .'
    print(new_sent)
