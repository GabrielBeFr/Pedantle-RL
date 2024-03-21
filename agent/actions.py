#### Make action with set of words (simple words & classical words)
#### Make action that look at the first unfound word (as the title is at the beginning of the article)

import numpy as np 
from collections import defaultdict
from gym_examples.wrappers.utils import filter_words
import re
from pdb import set_trace 
import agent
# from utils import get_nearest_words
from agent.utils import get_nearest_words
from sklearn.metrics.pairwise import cosine_similarity

# WORD_LIST_A = ["time","person","year","way","day", "thing", "man", "world", "life", "hand", "part", "child", "eye", "woman", "place", "work", "week", "case", "point", "government", "company", "number", "group", "problem", "fact","account","act","addition","adjustment","advertisement","agreement","air","amount","amusement","animal","answer","apparatus","approval","argument","art","attack","attempt","attention","attraction","authority","back","balance","base","behaviour","belief","birth","bit","bite","blood","blow","body","brass","bread","breath","brother","building","burn","burst","business","butter","canvas","care","cause","chalk","chance","change","cloth","coal","colour","comfort","committee","company","comparison","competition","condition","connection","control","cook","copper","copy","cork","cotton","cough","country","cover","crack","credit","crime","crush","cry","current","curve","damage","danger","daughter","day","death","debt","decision","degree","design","desire","destruction","detail","development","digestion","direction","discovery","discussion","disease","disgust","distance","distribution","division","doubt","drink","driving","dust","earth","edge","education","effect","end","error","event","example","exchange","existence","expansion","experience","expert","fact","fall","family","father","fear","feeling","fiction","field","fight","fire","flame","flight","flower","fold","food","force","form","friend","front","fruit","glass","gold","government","grain","grass","grip","group","growth","guide","harbour","harmony","hate","hearing","heat","help","history","hole","hope","hour","humour","ice","idea","impulse","increase","industry","ink","insect","instrument","insurance","interest","invention","iron","jelly","join","journey","judge","jump","kick","kiss","knowledge","land","language","laugh","law","lead","learning","leather","letter","level","lift","light","limit","linen","liquid","list","look","loss","love","machine","man","manager","mark","market","mass","meal","measure","meat","meeting","memory","metal","middle","milk","mind","mine","minute","mist","money","month","morning","mother","motion","mountain","move","music","name","nation","need","news","night","noise","note","number","observation","offer","oil","operation","opinion","order","organization","ornament","owner","page","pain","paint","paper","part","paste","payment","peace","person","place","plant","play","pleasure","point","poison","polish","porter","position","powder","power","price","print","process","produce","profit","property","prose","protest","pull","punishment","purpose","push","quality","question","rain","range","rate","ray","reaction","reading","reason","record","regret","relation","religion","representative","request","respect","rest","reward","rhythm","rice","river","road","roll","room","rub","rule","run","salt","sand","scale","science","sea","seat","secretary","selection","self","sense","servant","sex","shade","shake","shame","shock","side","sign","silk","silver","sister","size","sky","sleep","slip","slope","smash","smell","smile","smoke","sneeze","snow","soap","society","son","song","sort","sound","soup","space","stage","start","statement","steam","steel","step","stitch","stone","stop","story","stretch","structure","substance","sugar","suggestion","summer","support","surprise","swim","system","talk","taste","tax","teaching","tendency","test","theory","thing","thought","thunder","time","tin","top","touch","trade","transport","trick","trouble","turn","twist","unit","use","value","verse","vessel","view","voice","walk","war","wash","waste","water","wave","wax","way","weather","week","weight","wind","wine","winter","woman","wood","wool","word","work","wound","writing","year"]
WORD_LIST_A = ["time","month","Environment", "Agriculture", "Engineering", "Technology", "Education", "Health", "Finance", "Transport", "Communication", "Infrastructure", "Industry", "Science", "Research", "Business", "Government", "Energy", "Manufacturing", "Sustainability", "Development", "Policy", "Innovation", "Economy", "Climate", "Security", "Management", "Culture", "Resource", "Water", "Globalization", "Design"]

def first_word(observation, agent, logging):
    '''
    Return the closest word of the first word fitted in the article.
    '''
    model = agent.model
    i = 0
    while True:
        target_id = observation["index_of_words_to_find"][i]
        word = observation["fitted_words"][target_id]
        if word is not None:
            break
        if len(observation["index_of_words_to_find"]) == i-1:
            logging.info("Turned to random")
            return _random_word(observation, agent, logging)
        i += 1

    words, _ = get_nearest_words(agent, observation, target_id)

    for i, word in enumerate(words):
        word = re.split(r'[^a-zA-Z0-9]', word) # split the word to remove special characters (I'm -> ['I','m'])
        if word[0] not in observation["proposed_words"]:
            break
        # we have to add a second condition for words like "He'd" or "He's"
        elif len(word)>1 and word[1] not in observation["proposed_words"]: 
            break
    if i==len(words)-1:
        logging.info("Turned to random")
        return _random_word(observation, agent, logging)
    else:
        return word, target_id 

def list_classic_word(observation, agent, logging):
    '''
    Return the closest word of the first word fitted in the article.
    '''

    for i,word in enumerate(WORD_LIST_A):
        if word not in observation["proposed_words"]:
            break
    if i==len(WORD_LIST_A)-1:
        logging.info("Turned to random")
        return _random_word(observation, agent, logging)
    
    return [word], None 

def _random_word(observation, agent, logging):
    '''
    Return a random word from the model's vocabulary and None as a target as the coice is random.
    There is no strategy here.
    '''
    model = agent.model
    n = len(agent.voc)
    i = np.random.randint(0, n-1)
    word = model.index_to_key[i]

    return re.split(r'[^a-zA-Z0-9]', word), None # split the word to remove special characters (I'm -> ['I','m'])

def closest_word_of_random_word(observation, agent, logging):
    '''
    Return the closest word of a random word taken among the fitted words and the chosen word as the target.
    The idea is to explore rather than exploit.
    '''
    target_id = np.random.choice(
        a=observation["index_of_words_to_find"],
        size=1,
        )[0]

    if observation["fitted_words"][target_id] is None:
        logging.info("Turned to random")
        return _random_word(observation, agent, logging)
    
    words, _ = get_nearest_words(agent, observation, target_id)
    
    for i, word in enumerate(words):
        word = re.split(r'[^a-zA-Z0-9]', word)
        if word[0] not in observation["proposed_words"]:
            break
        # we have to add a second contion for words like "He'd" or "He's"
        elif len(word)>1 and word[1] not in observation["proposed_words"]: 
            break
    if i==len(words)-1:
        logging.info("Turned to random")
        return _random_word(observation, agent, logging)

    return word, target_id # split the word to remove special characters (I'm -> ['I','m'])

def closest_word_of_last_targetted_word(observation, agent, logging):
    '''
    Return the closest word of the word the fitted word that is in the position 
    of the last targetted word and the last targetted word as the target.
    The idea is to exploit rather than explore.
    '''
    last_target_id = agent.last_target_id

    # Let's first look at if the last target id corresponds to a fitting word. If not, we return a random word.
    if last_target_id is None or observation["fitted_words"][last_target_id] is None:
        logging.info("Turned to random")
        return _random_word(observation, agent, logging)

    words, _ = get_nearest_words(agent, observation, last_target_id)   

    for i,word in enumerate(words):
        word = re.split(r'[^a-zA-Z0-9]', word)
        if word[0] not in observation["proposed_words"]:
            break
        # we have to add a second contion for words like "He'd" or "He's"
        elif len(word)>1 and word[1] not in observation["proposed_words"]: 
            break
    if i==len(words)-1:
        logging.info("Turned to random")
        return _random_word(observation, agent, logging)

    return word, last_target_id # split the word to remove special characters (I'm -> ['I','m'])

def closest_of_closest_words(observation, agent, logging):
    '''
    Return the word with the highest similarity score
    among the closest words of each word fitted in the article
    and the corresponding word among the fitted ones as the target.
    '''    

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
        logging.info("Turned to random")
        return _random_word(observation, agent, logging)

    while True:
        closest_word = max(closest_words_score, key=lambda x: closest_words_score[x])
        target_id = closest_words_index[closest_word]
        word = re.split(r'[^a-zA-Z0-9]', closest_word)
        if word[0] not in observation["proposed_words"]:
            break
        elif len(word)>1 and word[1] not in observation["proposed_words"]:
            break
        closest_words_score.pop(closest_word)
        closest_words_index.pop(closest_word)
        if len(closest_words_score) == 0:
            logging.info("Turned to random")
            return _random_word(observation, agent, logging)

    return word, target_id # split the word to remove special characters (I'm -> ['I','m'])

ACTIONS = {
    "list_classic_word": list_classic_word,
    "first_word": first_word,
    "closest_word_of_random_word": closest_word_of_random_word,
    "closest_word_of_last_targetted_word": closest_word_of_last_targetted_word,
    "closest_of_closest_words": closest_of_closest_words,
}