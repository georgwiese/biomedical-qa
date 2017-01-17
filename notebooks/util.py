import math

import numpy as np
import matplotlib.pyplot as plt

def remove_negative_outliers(scores, threshold=-500):
    
    filtered_scores = scores[scores >= threshold]
    scores[scores < threshold] = filtered_scores.min()
    return scores

def text_heatmap(tokens, scores, token_highlight=None):
    """Displays a given text as a heatmap.

    Arguments:
        - tokens: List of token strings
        - scores: 1D numpy array of scores, one for each token
        - token_highlight: numpy array of booleans that
            specifies for each token whether it should be
            highlighted.
    """

    if token_highlight is None:
        token_highlight = np.zeros(len(tokens), dtype=np.bool)
        
    scores = remove_negative_outliers(scores)

    tokens_per_row = 10
    num_rows = math.ceil(len(tokens) / tokens_per_row)
    s_min, s_max = scores.min(), scores.max()

    f, axes = plt.subplots(num_rows, figsize=(15, num_rows / 2))
    for row in range(num_rows):

        axis = axes[row] if num_rows > 1 else axes

        axis.pcolor(scores[row * tokens_per_row:(row + 1) * tokens_per_row].reshape((1, -1)),
                    cmap="YlOrBr", vmin=s_min, vmax=s_max)

        axis.get_yaxis().set_visible(False)
        axis.get_xaxis().set_visible(False)

        for i in range(tokens_per_row):
            token_index = row * tokens_per_row + i
            token = tokens[token_index] if token_index < len(tokens) else ""
            is_highlighted = token_highlight[token_index] if token_index < len(tokens) else False
            color = "Green" if is_highlighted else "Black"
            axis.text(i + 0.5, 0.5, token,
                           color=color,
                           horizontalalignment='center',
                           verticalalignment='center')
    
    
def print_question(qa_setting, rev_vocab):
    question = [rev_vocab[w] for w in qa_setting.question]
    print("Question:")
    print(' '.join(question))
    print()

def print_answers(qa_setting, rev_vocab):
    answers = [' '.join([rev_vocab[w] for w in a]) for a in qa_setting.answers]
    print("Answers:")
    print(answers)
    print()

def print_predicted(qa_setting, answer_start, answer_end, rev_vocab):
    predicted =  " ".join([rev_vocab[i] for i in qa_setting.context[answer_start:answer_end+1]])
    print("Predicted:")
    print(predicted)
    print()

def print_context(qa_setting, rev_vocab):
    context = [rev_vocab[w] for w in qa_setting.context]
    print("Context:")
    print(' '.join(context))
    print()
    
    
def print_list(string_list):
    
    for s in string_list:
        print(" * " + str(s))
        
        
def maybe_flatten_list(l):
    
    if isinstance(l[0], str):
        l = [l]
    return [x for sublist in l for x in sublist]
    
    
def softmax(w):
    e = np.exp(w)
    dist = e / np.sum(e)
    return dist

def find_correct_tokens(qa_setting):
    """Returns a boolean np array of length len(tokens) that states if the token is part of an answer."""
    answers = [answer_option for answer in qa_setting.answers
                             for answer_option in answer] 
    tokens = [w for context in qa_setting.contexts
                for w in context]
    
    
    is_correct = np.zeros(len(tokens), dtype=np.bool)
    for i, start_token in enumerate(tokens):
        for answer in answers:
            if start_token == answer[0]: #and len(tokens) - i >= len(answer):
                is_correct[i] = True
                all_correct = True
                for j in range(1, len(answer)):
                    all_correct &= tokens[i + j] == answer[j]
                if all_correct:
                    for j in range(0, len(answer)):
                        is_correct[i + j] = True
    
    return is_correct