import numpy as np 
from collections import defaultdict
from gym_examples.wrappers.utils import filter_words
import re
from pdb import set_trace 
from agent.utils import get_nearest_words
from sklearn.metrics.pairwise import cosine_similarity

def _random_word(observation, agent, logging):
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

    target_id = np.random.choice(
        a=observation["index_of_words_to_find"],
        size=1,
        )[0]

    if observation["fitted_words"][target_id] is None:
        return _random_word(observation, agent, logging)
    
    words, _ = get_nearest_words(agent, observation, target_id)
    
    for word in words:
        word = re.split(r'[^a-zA-Z0-9]', word)
        if word[0] not in observation["proposed_words"]:
            break
        # we have to add a second contion for words like "He'd" or "He's"
        elif len(word)>1 and word[1] not in observation["proposed_words"]: 
            break

    ######TO-DO######
    # Sometimes, because of the uppercases, the word found is the
    # same as the random word. In this case, we may want to take
    # the second closest word.
    #################

    return word, target_id # split the word to remove special characters (I'm -> ['I','m'])

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
    last_target_id = agent.last_target_id

    # Let's first look at if the last target id corresponds to a fitting word. If not, we return a random word.
    if last_target_id is None or observation["fitted_words"][last_target_id] is None:
        return _random_word(observation, agent, logging)
    
    logging.info(f"Target memory: {last_target_id}")

    words, _ = get_nearest_words(agent, observation, last_target_id)   

    for word in words:
        word = re.split(r'[^a-zA-Z0-9]', word)
        if word[0] not in observation["proposed_words"]:
            break
        # we have to add a second contion for words like "He'd" or "He's"
        elif len(word)>1 and word[1] not in observation["proposed_words"]: 
            break
    ######TO-DO######
    # Sometimes, because of the uppercases, the word found is the
    # same as the random word. In this case, we may want to take
    # the second closest word.
    #################

    return word, last_target_id # split the word to remove special characters (I'm -> ['I','m'])

def closest_of_closest_words(observation, agent, logging):
    '''
    Return the word with the highest similarity score
    among the closest words of each word fitted in the article
    and the corresponding word among the fitted ones as the target.
    '''    
    model = agent.model

    closest_words_index = defaultdict(int)
    closest_words_score = defaultdict(int)
    turn_to_random = True
    for i in observation["index_of_words_to_find"]:
        word = observation["fitted_words"][i]
        if word is not None:
            turn_to_random = False
            word, score = get_nearest_words(agent, observation, i, n=1)
            word, score = word[0], score[0]
            closest_words_score[word] = score
            closest_words_index[word] = i

    if turn_to_random:
        return _random_word(observation, agent, logging)
        
    while True:
        closest_word = max(closest_words_score, key=lambda x: closest_words_score[x])
        target_id = closest_words_index[closest_word]
        word = re.split(r'[^a-zA-Z0-9]', closest_word)
        if word[0] not in observation["proposed_words"]:
            break
        elif len(word)>1 and word[1] not in observation["proposed_words"]:
            break
        else:
            closest_words_score.pop(closest_word)
            closest_words_index.pop(closest_word)
            if len(closest_words_score) == 0:
                return _random_word(observation, agent, logging)

    return word, target_id # split the word to remove special characters (I'm -> ['I','m'])

ACTIONS = {
    "random": _random_word,
    "closest_word_of_random_word": closest_word_of_random_word,
    "closest_word_of_last_targetted_word": closest_word_of_last_targetted_word,
    "closest_of_closest_words": closest_of_closest_words,
}