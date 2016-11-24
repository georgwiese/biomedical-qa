import math

import numpy as np
import matplotlib.pyplot as plt

def text_heatmap(tokens, scores, token_highlight=None):
    """Displays a given text as a heatmap.

    Arguments:
        - tokens: List of token strings
        - scores: numpy array of scores, one for each token
        - token_highlight: numpy array of booleans that
            specifies for each token whether it should be
            highlighted.
    """

    if token_highlight is None:
        token_highlight = np.zeros(len(tokens), dtype=np.bool)

    tokens_per_row = 10
    num_rows = math.ceil(len(tokens) / tokens_per_row)
    s_min, s_max = scores.min(), scores.max()

    f, axes = plt.subplots(num_rows, figsize=(15, num_rows / 2))
    for row in range(num_rows):

        axis = axes[row] if num_rows > 1 else axes

        axis.pcolor(scores[:, row * tokens_per_row:(row + 1) * tokens_per_row],
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
    
    
def print_question(batch, rev_vocab):
    question = [rev_vocab[w] for w in batch[0].question]
    print("Question:")
    print(' '.join(question))
    print()

def print_answers(batch, rev_vocab):
    answers = [' '.join([rev_vocab[w] for w in a]) for a in batch[0].answers]
    print("Answers:")
    print(answers)
    print()

def print_predicted(batch, answer_start, answer_end, rev_vocab):
    predicted =  " ".join([rev_vocab[i] for i in batch[0].context[answer_start:answer_end+1]])
    print("Predicted:")
    print(predicted)
    print()

def print_context(batch, rev_vocab):
    context = [rev_vocab[w] for w in batch[0].context]
    print("Context:")
    print(' '.join(context))
    print()
    
    
def softmax(w):
    e = np.exp(w)
    dist = e / np.sum(e)
    return dist

def find_correct_tokens(batch):
    """Returns a boolean np array of length len(tokens) that states if the token is part of an answer."""
    answers = batch[0].answers
    tokens = batch[0].context
    
    is_correct = np.zeros(len(tokens), dtype=np.bool)
    for i, start_token in enumerate(tokens):
        for answer in answers:
            if start_token == answer[0] and len(tokens) - i >= len(answer):
                all_correct = True
                for j in range(1, len(answer)):
                    all_correct &= tokens[i + j] == answer[j]
                if all_correct:
                    for j in range(0, len(answer)):
                        is_correct[i + j] = True
    
    return is_correct