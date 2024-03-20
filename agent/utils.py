

def get_nearest_words(agent, observation, target_id, n=10):
    model = agent.model
    index = agent.index
    target_word = observation["fitted_words"][target_id] 
    id = model.key_to_index[target_word]
    embedding = model.vectors[id]
    scores, words_index = index.search(embedding.reshape(1,-1), n+1) 
    words = [model.index_to_key[id] for id in words_index[0]]
    return words[1:], scores[0] # we remove the first word which is the target word