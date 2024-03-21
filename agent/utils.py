

def get_nearest_words(agent, observation, target_id, n=100):
    '''
    Return the n closest words to the target word in the model's vocabulary.
    Default is n = 10.

    params:
    - agent: an instance of the Agent class.
    - observation: a dictionary containing the current observation.
    - target_id: an integer representing the index of the target word in the article.
    - n: an integer representing the number of closest words to return.

    output:
    - words: a list of strings representing the n closest words to the target word.
    - scores: a list of floats representing the faiss scores for similarity
        between the target word and the n closest words.
    '''
    model = agent.model
    index = agent.index
    target_word = observation["fitted_words"][target_id] 
    id = model.key_to_index[target_word]
    embedding = model.vectors[id]
    scores, words_index = index.search(embedding.reshape(1,-1), n+1) 
    words = [model.index_to_key[id] for id in words_index[0]]
    return words[1:], scores[0] # we remove the first word which is the target word