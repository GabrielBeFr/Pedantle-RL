import numpy as np 
from collections import defaultdict
from gym_examples.wrappers.utils import filter_words
import re
from pdb import set_trace 

def random_word(observation, agent, logging):
    '''
    Return a random word from the model's vocabulary and None as a target as the coice is random.
    There is no strategy here.
    '''
    model = agent.model
    n = len(model.index_to_key)
    i = np.random.randint(0, n-1)
    word = model.index_to_key[i]

    return re.split(r'[^a-zA-Z0-9]', word), None # split the word to remove special characters (I'm -> ['I','m'])

def closest_word_of_random_word(observation, agent, logging):
    '''
    Return the closest word of a random word taken among the fitted words and the chosen word as the target.
    The idea is to explore rather than exploit.
    '''
    ######TO-DO######
    # - For now, the frequency of the words is taken into account to choose
    # the random word. It may be interesting to split this action into one
    # that does the same and one that chooses the random word with a uniform
    # distribution.
    # - Add a condition to avoid to choose a word that is already revealed
    # with a proximity of 1. 
    # - Take into account the words that have been tried unsuccessfully to
    # add them into the model.most_similar function as the 'negative' parameter.
    #################   
    model = agent.model
    words, frequencies = filter_words(observation, model)

    random_word = np.random.choice(
        a=words,
        size=1,
        p=frequencies/np.sum(frequencies),
        )[0]
    
    target_id = observation["fitted_words"].index(random_word)
    
    try:
        word = model.most_similar(
            positive=agent.pos_neg_words[target_id]["positive"],
            negative=agent.pos_neg_words[target_id]["negative"],
            topn=1,
            )[0][0]
    except:
        logging.info("Action turned into Random.")
        return random_word(observation, agent, logging)
    ######TO-DO######
    # Sometimes, because of the uppercases, the word found is the
    # same as the random word. In this case, we may want to take
    # the second closest word.
    #################

    return re.split(r'[^a-zA-Z0-9]', word), target_id # split the word to remove special characters (I'm -> ['I','m'])

def closest_word_of_last_targetted_word(observation, agent, logging):
    '''
    Return the closest word of the word the fitted word that is in the position 
    of the last targetted word and the last targetted word as the target.
    The idea is to exploit rather than explore.
    '''
    ######TO-DO######
    # For now, we get the targetted word by looking at the last
    # word that was proposed. However, for humans don't do this,
    # they target a word in the article and try to find it by looking 
    # at closest word found yet. We may want to implement the strategy
    # this way.
    #################
    model = agent.model
    last_target_id = agent.last_target_id
    if last_target_id is None or observation["fitted_words"][last_target_id] is None:
        return random_word(observation, agent, logging)
    logging.info(f"Target memory: {last_target_id}")
    try:
        word = model.most_similar(
            positive=agent.pos_neg_words[last_target_id]["positive"],
            negative=agent.pos_neg_words[last_target_id]["negative"],
            topn=1,
            )[0][0]
    except:
        logging.info("Action turned into Random.")
        return random_word(observation, agent, logging)
    ######TO-DO######
    # Sometimes, because of the uppercases, the word found is the
    # same as the random word. In this case, we may want to take
    # the second closest word.
    #################

    return re.split(r'[^a-zA-Z0-9]', word), last_target_id # split the word to remove special characters (I'm -> ['I','m'])

def closest_of_closest_words(observation, agent, logging):
    '''
    Return the word with the highest similarity score
    among the closest words of each word fitted in the article
    and the corresponding word among the fitted ones as the target.
    '''    
    model = agent.model
    closest_of_words_index = defaultdict(int)
    closest_words_score = defaultdict(int)
    try:
        for i, word in enumerate(observation["fitted_words"]):
            if word is not None and re.match(r'^[a-zA-Z0-9]+$', word):
                closest_word, score = model.most_similar(
                    positive=agent.pos_neg_words[i]["positive"], 
                    negative=agent.pos_neg_words[i]["negative"],
                    topn=1,
                    )[0]
                closest_words_score[closest_word] = score
                closest_of_words_index[closest_word] = i
            elif i == len(observation["fitted_words"])-1:
                return random_word(observation, agent, logging)

        word = max(closest_words_score, key=lambda x: closest_words_score[x])
        target_id = closest_of_closest_words[word]
    except:
        logging.info("Action turned into Random.")
        return random_word(observation, agent, logging), None

    return re.split(r'[^a-zA-Z0-9]', word), target_id # split the word to remove special characters (I'm -> ['I','m'])

ACTIONS = {
    "random": random_word,
    "closest_word_of_random_word": closest_word_of_random_word,
    "closest_word_of_last_targetted_word": closest_word_of_last_targetted_word,
    "closest_of_closest_words": closest_of_closest_words,
}