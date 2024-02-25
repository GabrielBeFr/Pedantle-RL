# This file contains utility functions that are used in the environment.

import re

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

    output_string = re.sub(r'\s+', ' ', output_string)
    words = re.findall(r"[\w]+|[.,!?;-_=+\(\)\[\]/']+", output_string)
    return words[:max_length]

def process_title(title):
    import re
    title = re.sub(r'\s+', ' ', title)
    words = re.findall(r"[\w']+|[.,!?;-_=+\(\)\[\]/']", title)
    return words

def load_wiki_page():
    import pandas as pd

    wiki = pd.read_csv("/home/gabriel/cours/RL/projet/wikipedia_simple.csv")
    article = wiki.sample()
    return article.to_dict(orient="records")[0]