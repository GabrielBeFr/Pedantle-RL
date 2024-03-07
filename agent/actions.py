import numpy as np 
from collections import defaultdict
from gym_examples.wrappers.utils import filter_words

def random_word(observation, model, target_memory, logging):
    '''
    Return a random word from the model's vocabulary and None as a target as the coice is random.
    There is no strategy here.
    '''
    n = len(model.index_to_key)
    i = np.random.randint(0, n-1)
    word = model.index_to_key[i]

    return word.split('_'), None

def closest_word_of_random_word(observation, model, target_memory, logging):
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
    words, frequencies = filter_words(observation, model)

    random_word = np.random.choice(
        a=words,
        size=1,
        p=frequencies/np.sum(frequencies),
        )[0]
    
    word = model.most_similar(random_word, topn=1)[0][0]
    ######TO-DO######
    # Sometimes, because of the uppercases, the word found is the
    # same as the random word. In this case, we may want to take
    # the second closest word.
    #################

    return word.split('_'), random_word

def closest_word_of_last_targetted_word(observation, model, target_memory, logging):
    '''
    Return the closest word of the word that was proposed last and the last targetted word as the target.
    The idea is to exploit rather than explore.
    '''
    ######TO-DO######
    # For now, we get the targetted word by looking at the last
    # word that was proposed. However, for humans don't do this,
    # they target a word in the article and try to find it by looking 
    # at closest word found yet. We may want to implement the strategy
    # this way.
    #################
    proposed_words = observation["proposed_words"]
    if target_memory[-1] is None:
        word = np.random.choice(model.index_to_key)
    else:
        word = model.most_similar(target_memory[-1], topn=1)[0][0]
    ######TO-DO######
    # Sometimes, because of the uppercases, the word found is the
    # same as the random word. In this case, we may want to take
    # the second closest word.
    #################

    return word.split('_'), target_memory[-1]

def closest_of_closest_words(observation, model, target_memory, logging):
    '''
    Return the word with the highest similarity score
    among the closest words of each word fitted in the article
    and the corresponding word among the fitted ones as the target.
    '''    
    fitted_words, _ = filter_words(observation, model)

    closest_words_score = defaultdict(int)
    closest_words_words = defaultdict(str)
    for word in fitted_words:
        closest_word, score = model.most_similar(word, topn=1)[0]
        closest_words_words[closest_word] = word
        closest_words_score[closest_word] += score

    closest_word = max(closest_words_score, key=lambda x: closest_words_score[x])
    ######TO-DO######
    # Sometimes, because of the uppercases, the word found is the
    # same as the random word. In this case, we may want to take
    # the second closest word.
    #################

    return closest_word.split('_'), closest_words_words[closest_word]

ACTIONS = {
    "random": random_word,
    "closest_word_of_random_word": closest_word_of_random_word,
    "closest_word_of_last_targetted_word": closest_word_of_last_targetted_word,
    "closest_of_closest_words": closest_of_closest_words,
}