# This file contains utility functions that are used in the environment.
import pandas as pd
import re
from collections import defaultdict
import numpy as np

def process_article(article, max_length):
    '''
    This function takes an article as input and returns a cleaned list of words.

    params:
    - article: a string representing the article.
    - max_length: an integer representing the maximum length of the list of words.

    output:
    - words: a list of words of maximum length max_length.
    '''
    
    input_string = article
    pattern = r'([\s\S]*?)\n\n\w+[\s]*\n\n'
    match = re.search(pattern, input_string)

    if match:
        text_before_word = match.group(1)
        output_string = text_before_word
    else:
        output_string = input_string

    output_string = re.sub(r'\s+', ' ', output_string).lower()
    words = re.findall(r"[\w]+|[.,!?;-_=+\(\)\[\]/']+", output_string)
    return words[:max_length]

def process_title(title):
    title = re.sub(r'\s+', ' ', title)
    words = re.findall(r"[\w']+|[.,!?;-_=+\(\)\[\]/']", title)
    return words

def load_wiki_page(wiki_file):
    wiki = pd.read_csv(wiki_file)
    article = wiki.sample()
    return article.to_dict(orient="records")[0]

def filter_words(observation, voc):
    fitted_words = np.array(observation["fitted_words"])[np.where(observation["words_prox"] != 1)[0]]
    sequence_of_words = set(fitted_words)
    filtered_words = defaultdict(int)
    for word in sequence_of_words:
        if word is not None and re.match(r'^[a-zA-Z0-9]+$', word):
            if word in voc:
                filtered_words[word] += 1

    frequencies = np.array(list(filtered_words.values()))
    words = np.array(list(filtered_words.keys()))
    return words, frequencies