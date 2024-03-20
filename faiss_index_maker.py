import numpy as np 
import faiss
from gym_examples.wrappers.sim_computer import load_embedding_model

# Making test index
print("Making test index")
print("Initializing index")
nlist = 100 #?
m = 10 # number of bytes per vector
d = 300 # dimension of the vectors
quantizer = faiss.IndexFlatL2(d) # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

print("Loading model")
model = load_embedding_model(True)
vectors = model.vectors

print("Training index")
index.train(vectors)
print("Adding vectors to index")
index.add_with_ids(vectors,np.arange(vectors.shape[0]))
faiss.write_index(index, "data/word2vec_test.faiss")

# Making full index
print("Making full index")
print("Initializing index")
nlist = 100 #?
m = 10 # number of bytes per vector
d = 300 # dimension of the vectors
quantizer = faiss.IndexFlatL2(d) # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

print("Loading model")
model = load_embedding_model(False)
vectors = model.vectors

print("Training index")
index.train(vectors)
print("Adding vectors to index")
index.add_with_ids(vectors,np.arange(vectors.shape[0]))
faiss.write_index(index, "data/word2vec_test.faiss")