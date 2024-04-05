import numpy as np 
import faiss
from gym_examples.wrappers.sim_computer import load_embedding_model
from sklearn.preprocessing import normalize

# Making test index
print("Making test index")
print("Initializing index")
d = 300 # dimension of the vectors

index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 256
index.hnsw.efSearch = 256
index = faiss.IndexRefineFlat(index)
index = faiss.IndexIDMap(index)

print("Loading model")
model, _ = load_embedding_model(True)
vectors = model.vectors
vectors = normalize(vectors)

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
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 256
index.hnsw.efSearch = 256
index = faiss.IndexRefineFlat(index)
index = faiss.IndexIDMap(index)

print("Loading model")
model, _ = load_embedding_model(False)
vectors = model.vectors
vectors = normalize(vectors)

print("Training index")
index.train(vectors)
print("Adding vectors to index")
index.add_with_ids(vectors,np.arange(vectors.shape[0]))
faiss.write_index(index, "data/v3_cpu_word2vec_full.faiss")

# v1 : index = faiss.index_factory(300, 'IVF65536_HNSW32,Flat,RFlat')
# index = faiss.IndexIDMap(index)

# v2 : index = faiss.IndexHNSWFlat(d, 32)
# index.hnsw.efConstruction = 80
# index.hnsw.efSearch = 64
# index = faiss.IndexIDMap(index)

# v3: vectors = normalize(vectors)
# index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
# index.hnsw.efConstruction = 256
# index.hnsw.efSearch = 256
# index = faiss.IndexRefineFlat(index)
# index = faiss.IndexIDMap(index)