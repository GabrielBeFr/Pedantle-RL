import numpy as np 
import faiss
from gym_examples.wrappers.sim_computer import load_embedding_model

# Making test index
print("Making test index")
print("Initializing index")
d = 300 # dimension of the vectors
index = faiss.index_factory(300, 'IVF1000_HNSW32,PQ30,RFlat')
index = faiss.IndexIDMap(index)

print("Loading model")
model, _ = load_embedding_model(True)
vectors = model.vectors

print("Training index")
index.train(vectors)
print("Adding vectors to index")
index.add_with_ids(vectors,np.arange(vectors.shape[0]))
faiss.write_index(index, "data/cpu_word2vec_test.faiss")

# Making full index
print("Making full index")
print("Initializing index")
nlist = 100 #?
m = 10 # number of bytes per vector
d = 300 # dimension of the vectors
index = faiss.index_factory(300, 'IVF65536_HNSW32,PQ30,RFlat')
index = faiss.IndexIDMap(index)

print("Loading model")
model, _ = load_embedding_model(False)
vectors = model.vectors

print("Training index")
index.train(vectors)
print("Adding vectors to index")
index.add_with_ids(vectors,np.arange(vectors.shape[0]))
faiss.write_index(index, "data/cpu_word2vec_test.faiss")