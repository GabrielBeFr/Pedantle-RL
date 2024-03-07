import numpy as np 
from collections import defaultdict
from gym_examples.wrappers.utils import filter_words

def random_word(observation, model, logging):
    '''
    Return a random word from the model's vocabulary.
    There is no strategy here.
    '''
    n = len(model.index_to_key)
    i = np.random.randint(0, n-1)
    word = model.index_to_key[i]

    logging.info(f"Random word chosen from the vocabulary: {word}")

    return word.split('_')

def closest_word_of_random_word(observation, model, logging):
    '''
    Return the closest word of a random word taken among the fitted words.
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
    fitted_words = set(observation["fitted_words"])
    
    words, frequencies = filter_words(fitted_words, model)

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

    logging.info(f"Random word chosen from the fitted words in the article: {random_word} and the most similar word found is: {word}")

    return word.split('_')

def closest_word_of_last_targetted_word(observation, model, logging):
    '''
    Return the closest word of the word that was proposed last.
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
    word = model.most_similar(proposed_words[-1], topn=1)[0][0]
    ######TO-DO######
    # Sometimes, because of the uppercases, the word found is the
    # same as the random word. In this case, we may want to take
    # the second closest word.
    #################

    logging.info(f"Last proposed word: {proposed_words[-1]} and the most similar word found is: {word}")

    return word.split('_')

def closest_of_closest_words(observation, model, logging):
    '''
    Return the word with the highest similarity score
    among the closest words of each word fitted in the article.
    '''
    fitted_words = set(observation["fitted_words"])
    
    fitted_words, _ = filter_words(fitted_words, model)

    closest_words = defaultdict(int)
    for word in fitted_words:
        closest_word, score = model.most_similar(word, topn=1)[0]
        closest_words[closest_word] += score

    closest_word = max(closest_words, key=lambda x: closest_words[x])
    ######TO-DO######
    # Sometimes, because of the uppercases, the word found is the
    # same as the random word. In this case, we may want to take
    # the second closest word.
    #################

    logging.info(f"The closest of the closest words is {closest_word} with a score of {closest_words[closest_word]}.")

    return closest_word.split('_')

ACTIONS = {
    "random": random_word,
    "closest_word_of_random_word": closest_word_of_random_word,
    "closest_word_of_last_targetted_word": closest_word_of_last_targetted_word,
    "closest_of_closest_words": closest_of_closest_words,
}