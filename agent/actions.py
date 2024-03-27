#### Make action with set of words (simple words & classical words)
#### Make action that look at the first unfound word (as the title is at the beginning of the article)

import numpy as np 
from collections import defaultdict
from gym_examples.wrappers.utils import filter_words
import re
from pdb import set_trace 
import agent
from agent.utils import get_nearest_words
from sklearn.metrics.pairwise import cosine_similarity
import random

WORD_LIST_RANDOM = {"person","year","way","day", "thing", "man", "world", "life", "hand", "part", "child", "eye", "woman", "place", "work", "week", "case", "point", "government", "company", "number", "group", "problem", "fact","account","act","addition","adjustment","advertisement","agreement","air","amount","amusement","animal","answer","apparatus","approval","argument","art","attack","attempt","attention","attraction","authority","back","balance","base","behaviour","belief","birth","bit","bite","blood","blow","body","brass","bread","breath","brother","building","burn","burst","business","butter","canvas","care","cause","chalk","chance","change","cloth","coal","colour","comfort","committee","company","comparison","competition","condition","connection","control","cook","copper","copy","cork","cotton","cough","country","cover","crack","credit","crime","crush","cry","current","curve","damage","danger","daughter","day","death","debt","decision","degree","design","desire","destruction","detail","development","digestion","direction","discovery","discussion","disease","disgust","distance","distribution","division","doubt","drink","driving","dust","earth","edge","education","effect","end","error","event","example","exchange","existence","expansion","experience","expert","fact","fall","family","father","fear","feeling","fiction","field","fight","fire","flame","flight","flower","fold","food","force","form","friend","front","fruit","glass","gold","government","grain","grass","grip","group","growth","guide","harbour","harmony","hate","hearing","heat","help","history","hole","hope","hour","humour","ice","idea","impulse","increase","industry","ink","insect","instrument","insurance","interest","invention","iron","jelly","join","journey","judge","jump","kick","kiss","knowledge","land","language","laugh","law","lead","learning","leather","letter","level","lift","light","limit","linen","liquid","list","look","loss","love","machine","man","manager","mark","market","mass","meal","measure","meat","meeting","memory","metal","middle","milk","mind","mine","minute","mist","money","month","morning","mother","motion","mountain","move","music","name","nation","need","news","night","noise","note","number","observation","offer","oil","operation","opinion","order","organization","ornament","owner","page","pain","paint","paper","part","paste","payment","peace","person","place","plant","play","pleasure","point","poison","polish","porter","position","powder","power","price","print","process","produce","profit","property","prose","protest","pull","punishment","purpose","push","quality","question","rain","range","rate","ray","reaction","reading","reason","record","regret","relation","religion","representative","request","respect","rest","reward","rhythm","rice","river","road","roll","room","rub","rule","run","salt","sand","scale","science","sea","seat","secretary","selection","self","sense","servant","sex","shade","shake","shame","shock","side","sign","silk","silver","sister","size","sky","sleep","slip","slope","smash","smell","smile","smoke","sneeze","snow","soap","society","son","song","sort","sound","soup","space","stage","start","statement","steam","steel","step","stitch","stone","stop","story","stretch","structure","substance","sugar","suggestion","summer","support","surprise","swim","system","talk","taste","tax","teaching","tendency","test","theory","thing","thought","thunder","time","tin","top","touch","trade","transport","trick","trouble","turn","twist","unit","use","value","verse","vessel","view","voice","walk","war","wash","waste","water","wave","wax","way","weather","week","weight","wind","wine","winter","woman","wood","wool","word","work","wound","writing","year"}
WORD_LIST_USUAL = {"environment", "agriculture", "engineering", "technology", "education", "health", "finance", "transport", "communication", "infrastructure", "industry", "science", "research", "business", "government", "energy", "manufacturing", "sustainability", "development", "policy", "innovation", "economy", "climate", "security", "management", "culture", "resource", "water", "globalization", "design"}
WORDS_BIG_LIST = {"the","of","to","and","a","in","is","it","you","that","he","was","for","on","are","with","as","I","his","they","be","at","one","have","this","from","or","had","by","not","word","but","what","some","we","can","out","other","were","all","there","when","up","use","your","how","said","an","each","she","which","do","their","time","if","will","way","about","many","then","them","write","would","like","so","these","her","long","make","thing","see","him","two","has","look","more","day","could","go","come","did","number","sound","no","most","people","my","over","know","water","than","call","first","who","may","down","side","been","now","find","any","new","work","part","take","get","place","made","live","where","after","back","little","only","round","man","year","came","show","every","good","me","give","our","under","name","very","through","just","form","sentence","great","think","say","help","low","line","differ","turn","cause","much","mean","before","move","right","boy","old","too","same","tell","does","set","three","want","air","well","also","play","small","end","put","home","read","hand","port","large","spell","add","even","land","here","must","big","high","such","follow","act","why","ask","men","change","went","light","kind","off","need","house","picture","try","us","again","animal","point","mother","world","near","build","self","earth","father","head","stand","own","page","should","country","found","answer","school","grow","study","still","learn","plant","cover","food","sun","four","between","state","keep","eye","never","last","let","thought","city","tree","cross","farm","hard","start","might","story","saw","far","sea","draw","left","late","run","don't","while","press","close","night","real","life","few","north","open","seem","together","next","white","children","begin","got","walk","example","ease","paper","group","always","music","those","both","mark","often","letter","until","mile","river","car","feet","care","second","book","carry","took","science","eat","room","friend","began","idea","fish","mountain","stop","once","base","hear","horse","cut","sure","watch","color","face","wood","main","enough","plain","girl","usual","young","ready","above","ever","red","list","though","feel","talk","bird","soon","body","dog","family","direct","pose","leave","song","measure","door","product","black","short","numeral","class","wind","question","happen","complete","ship","area","half","rock","order","fire","south","problem","piece","told","knew","pass","since","top","whole","king","space","heard","best","hour","better","true","during","hundred","five","remember","step","early","hold","west","ground","interest","reach","fast","verb","sing","listen","six","table","travel","less","morning","ten","simple","several","vowel","toward","war","lay","against","pattern","slow","center","love","person","money","serve","appear","road","map","rain","rule","govern","pull","cold","notice","voice","unit","power","town","fine","certain","fly","fall","lead","cry","dark","machine","note","wait","plan","figure","star","box","noun","field","rest","correct","able","pound","done","beauty","drive","stood","contain","front","teach","week","final","gave","green","oh","quick","develop","ocean","warm","free","minute","strong","special","mind","behind","clear","tail","produce","fact","street","inch","multiply","nothing","course","stay","wheel","full","force","blue","object","decide","surface","deep","moon","island","foot","system","busy","test","record","boat","common","gold","possible","plane","stead","dry","wonder","laugh","thousand","ago","ran","check","game","shape","equate","hot","miss","brought","heat","snow","tire","bring","yes","distant","fill","east","paint","language","among","grand","ball","yet","wave","drop","heart","am","present","heavy","dance","engine","position","arm","wide","sail","material","size","vary","settle","speak","weight","general","ice","matter","circle","pair","include","divide","syllable","felt","perhaps","pick","sudden","count","square","reason","length","represent","art","subject","region","energy","hunt","probable","bed","brother","egg","ride","cell","believe","fraction","forest","sit","race","window","store","summer","train","sleep","prove","lone","leg","exercise","wall","catch","mount","wish","sky","board","joy","winter","sat","written","wild","instrument","kept","glass","grass","cow","job","edge","sign","visit","past","soft","fun","bright","gas","weather","month","million","bear","finish","happy","hope","flower","clothe","strange","gone","jump","baby","eight","village","meet","root","buy","raise","solve","metal","whether","push","seven","paragraph","third","shall","held","hair","describe","cook","floor","either","result","burn","hill","safe","cat","century","consider","type","law","bit","coast","copy","phrase","silent","tall","sand","soil","roll","temperature","finger","industry","value","fight","lie","beat","excite","natural","view","sense","ear","else","quite","broke","case","middle","kill","son","lake","moment","scale","loud","spring","observe","child","straight","consonant","nation","dictionary","milk","speed","method","organ","pay","age","section","dress","cloud","surprise","quiet","stone","tiny","climb","cool","design","poor","lot","experiment","bottom","key","iron","single","stick","flat","twenty","skin","smile","crease","hole","trade","melody","trip","office","receive","row","mouth","exact","symbol","die","least","trouble","shout","except","wrote","seed","tone","join","suggest","clean","break","lady","yard","rise","bad","blow","oil","blood","touch","grew","cent","mix","team","wire","cost","lost","brown","wear","garden","equal","sent","choose","fell","fit","flow","fair","bank","collect","save","control","decimal","gentle","woman","captain","practice","separate","difficult","doctor","please","protect","noon","whose","locate","ring","character","insect","caught","period","indicate","radio","spoke","atom","human","history","effect","electric","expect","crop","modern","element","hit","student","corner","party","supply","bone","rail","imagine","provide","agree","thus","capital","won't","chair","danger","fruit","rich","thick","soldier","process","operate","guess","necessary","sharp","wing","create","neighbor","wash","bat","rather","crowd","corn","compare","poem","string","bell","depend","meat","rub","tube","famous","dollar","stream","fear","sight","thin","triangle","planet","hurry","chief","colony","clock","mine","tie","enter","major","fresh","search","send","yellow","gun","allow","print","dead","spot","desert","suit","current","lift","rose","continue","block","chart","hat","sell","success","company","subtract","event","particular","deal","swim","term","opposite","wife","shoe","shoulder","spread","arrange","camp","invent","cotton","born","determine","quart","nine","truck","noise","level","chance","gather","shop","stretch","throw","shine","property","column","molecule","select","wrong","gray","repeat","require","broad","prepare","salt","nose","plural","anger","claim","continent","oxygen","sugar","death","pretty","skill","women","season","solution","magnet","silver","thank","branch","match","suffix","especially","fig","afraid","huge","sister","steel","discuss","forward","similar","guide","experience","score","apple","bought","led","pitch","coat","mass","card","band","rope","slip","win","dream","evening","condition","feed","tool","total","basic","smell","valley","nor","double","seat","arrive","master","track","parent","shore","division","sheet","substance","favor","connect","post","spend","chord","fat","glad","original","share","station","dad","bread","charge","proper","bar","offer","segment","slave","duck","instant","market","degree","populate","chick","dear","enemy","reply","drink","occur","support","speech","nature","range","steam","motion","path","liquid","log","meant","quotient","teeth","shell","neck"}

def first_word(observation, agent, logging):
    '''
    Return the closest word of the first word fitted in the article.
    '''
    model = agent.model
    target_id = 0
    while True:
        word = observation["fitted_words"][target_id]
        if word is not None and re.match(r'^[a-zA-Z0-9]+$', word):
            if random.random() < 0.8:
                break
        if len(observation["fitted_words"])-1 == target_id:
            logging.info("Turned to full random")
            return _random_word(observation, agent, logging)
        target_id += 1

    try:
        words, _ = get_nearest_words(agent, observation, target_id, n=200)
    except:
        logging.info("Target word not in voc, turned to full random")
        return _random_word(observation, agent, logging)

    for i, word in enumerate(words):
        word = re.split(r'[^a-zA-Z0-9]', word) # split the word to remove special characters (I'm -> ['I','m'])
        if word[0].lower() not in observation["proposed_words"] and not word[0]=='':
            break
        # we have to add a second condition for words like "He'd" or "He's"
        elif len(word)>1 and word[1].lower() not in observation["proposed_words"] and not word[1]=='': 
            break
        if i==len(words)-1:
            logging.info("Turned to usual word")
            return list_classic_word(observation, agent, logging)
    return [item.lower() for item in word if item != ""], target_id 

def list_classic_word(observation, agent, logging):
    '''
    Return the closest word of the first word fitted in the article.
    '''
    try:
        unproposed_classics = WORD_LIST_USUAL - set(observation["proposed_words"])
    except:
        logging.info(f'type(observation["proposed_words"]): {type(observation["proposed_words"])}')

    if len(unproposed_classics)==0:
        unproposed_random = WORD_LIST_RANDOM - set(observation["proposed_words"])

        if len(unproposed_random)==0:
            unproposed_words = WORDS_BIG_LIST - set(observation["proposed_words"])

            if len(unproposed_words)==0:
                logging.info("No more word in the LISTS, turned to full random.")
                return _random_word(observation, agent, logging)
            
            else:
                word = np.random.choice(list(unproposed_words), size=1)
                return [word.lower()], None

        for word in unproposed_random:
            return [word.lower()], None

    for word in unproposed_classics:
        return [word.lower()], None 

def _random_word(observation, agent, logging):
    '''
    Return a random word from the model's vocabulary and None as a target as the coice is random.
    There is no strategy here.
    '''
    model = agent.model

    n = len(agent.voc)
    i = np.random.randint(0, n-1)
    word = model.index_to_key[i]
    word = re.split(r'[^a-zA-Z0-9]', word)
    return [item.lower() for item in word], None # split the word to remove special characters (I'm -> ['I','m'])

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
        logging.info("Turned to full random")
        return _random_word(observation, agent, logging)
    
    try:
        words, _ = get_nearest_words(agent, observation, target_id, n=200)
    except:
        logging.info("Target word not in voc, turned to full random")
        return _random_word(observation, agent, logging)
    
    for i, word in enumerate(words):
        word = re.split(r'[^a-zA-Z0-9]', word)
        if word[0].lower() not in observation["proposed_words"] and not word[0]=='':
            break
        # we have to add a second contion for words like "He'd" or "He's"
        elif len(word)>1 and word[1].lower() not in observation["proposed_words"] and not word[1]=='': 
            break
        if i==len(words)-1:
            logging.info("Turned to usual word")
            return list_classic_word(observation, agent, logging)

    return [item.lower() for item in word if item != ""], target_id # split the word to remove special characters (I'm -> ['I','m'])

def closest_word_of_last_targetted_word(observation, agent, logging):
    '''
    Return the closest word of the word the fitted word that is in the position 
    of the last targetted word and the last targetted word as the target.
    The idea is to exploit rather than explore.
    '''
    last_target_id = agent.last_target_id

    # Let's first look at if the last target id corresponds to a fitting word. If not, we return a random word.
    if last_target_id is None or observation["fitted_words"][last_target_id] is None:
        logging.info("Turned to full random")
        return _random_word(observation, agent, logging)

    try:
        words, _ = get_nearest_words(agent, observation, last_target_id, n=200)   
    except:
        logging.info("Target word not in voc, turned to full random")
        return _random_word(observation, agent, logging)

    for i,word in enumerate(words):
        word = re.split(r'[^a-zA-Z0-9]', word)
        if word[0].lower() not in observation["proposed_words"] and not word[0]=='':
            break
        # we have to add a second contion for words like "He'd" or "He's"
        elif len(word)>1 and word[1].lower() not in observation["proposed_words"] and not word[1]=='': 
            break
        if i==len(words)-1:
            logging.info("Turned to usual word")
            return list_classic_word(observation, agent, logging)
    logging.info(f"Raw word: {word}")
    return [item.lower() for item in word if item != ""], last_target_id # split the word to remove special characters (I'm -> ['I','m'])

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
        logging.info("Turned to full random")
        return _random_word(observation, agent, logging)

    while True:
        closest_word = max(closest_words_score, key=lambda x: closest_words_score[x])
        target_id = closest_words_index[closest_word]
        word = re.split(r'[^a-zA-Z0-9]', closest_word)
        if word[0].lower() not in observation["proposed_words"] and not word[0]=='':
            break
        elif len(word)>1 and word[1].lower() not in observation["proposed_words"] and not word[1]=='':
            break
        closest_words_score.pop(closest_word)
        closest_words_index.pop(closest_word)
        if len(closest_words_score) == 0:
            logging.info("Turned to full random")
            return _random_word(observation, agent, logging)

    return [item.lower() for item in word if item != ""], target_id # split the word to remove special characters (I'm -> ['I','m'])

def best_fitted_word(observation, agent, logging):
    '''
    The target word is the word whose observation["words_prox"] is the highest.
    It returns the nearest word to the target word, and the target id.
    '''
    intermed = np.argmax(observation["words_prox"][observation["index_of_words_to_find"]])
    target_id = observation["index_of_words_to_find"][intermed]
    word = observation["fitted_words"][target_id]
    if word is None:
        logging.info("Turned to full random")
        return _random_word(observation, agent, logging)
    
    try:
        words, _ = get_nearest_words(agent, observation, target_id, n=200)
    except:
        logging.info("Target word not in voc, turned to full random")
        return _random_word(observation, agent, logging)
    
    for i, word in enumerate(words):
        word = re.split(r'[^a-zA-Z0-9]', word)
        if word[0].lower() not in observation["proposed_words"] and not word[0]=='':
            break
        # we have to add a second contion for words like "He'd" or "He's"
        elif len(word)>1 and word[1].lower() not in observation["proposed_words"] and not word[1]=='': 
            break
        if i==len(words)-1:
            logging.info("Turned to usual word")
            return list_classic_word(observation, agent, logging)
    return [item.lower() for item in word if item != ""], target_id # split the word to remove special characters (I'm -> ['I','m'])

def look_for_title(observation, agent, logging):
    '''
    The target word is, randomly, one of the word of the title if it has been found.
    It returns the nearest word to the target word, and the target id.
    '''
    if all(word is None for word in observation["words_title"]):
        logging.info("Empty title, turned to full random")
        return _random_word(observation, agent, logging)
    word_title = random.choice(observation["words_title"])
    word = word_title.lower()
    if word is None:
        logging.info("Full title is None, turned to full random")
        return _random_word(observation, agent, logging)
    logging.info(f"Title word: {word}")
    
    try:
        words, _ = get_nearest_words(agent, observation, target_id=None, direct_word=word, n=200)
    except:
        logging.info("Target word not in voc, turned to full random")
        return _random_word(observation, agent, logging)

    for i, word in enumerate(words):
        word = re.split(r'[^a-zA-Z0-9]', word)
        if word[0].lower() not in observation["proposed_words"] and not word[0]=='':
            break
        # we have to add a second contion for words like "He'd" or "He's"
        elif len(word)>1 and word[1].lower() not in observation["proposed_words"] and not word[1]=='': 
            break
        if i==len(words)-1:
            logging.info("Turned to usual word")
            return list_classic_word(observation, agent, logging)
    return [item.lower() for item in word if item != ""], None # split the word to remove special characters (I'm -> ['I','m'])

ACTIONS = {
    "list_classic_word": list_classic_word,
    "first_word": first_word,
    "closest_word_of_random_word": closest_word_of_random_word,
    "closest_word_of_last_targetted_word": closest_word_of_last_targetted_word,
    "closest_of_closest_words": closest_of_closest_words,
    "best_fitted_word": best_fitted_word,
    "look_for_title": look_for_title,
}