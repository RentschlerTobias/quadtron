
from tokenizer import Tokenizer2D
from embedding import Embedding
import torch

path = '../meshtron_autoregession/data/quad_meshes.pt'
meshes = torch.load(path)
mesh = meshes[0]

quantization = 512
d_model = 128
vertices = mesh.x[:, 0:2]
faces = mesh.faces
tokenizer = Tokenizer2D(quantization_levels=quantization)
tokens, info = tokenizer.tokenize(vertices, faces, verbose=False)
token_sequence = len(tokens)

embedder = Embedding(vocab_size=quantization+3,
                     d_model=d_model, max_len=token_sequence)
embeddings = embedder(torch.tensor(tokens))
embeddings.size()


for i in range(10):
    print(i)
