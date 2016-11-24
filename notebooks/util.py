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
