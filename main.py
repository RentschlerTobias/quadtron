
from tokenizer import Tokenizer2D
from embedding import Embedding
from hourglass_transformer import HourglassTransformer
import torch
from point_encoder import PerceiverPointEncoder
path = '../data/quad_data.pt'
meshes = torch.load(path)

quantization = 1024
d_model = 512
n_latents = d_model
input_dim = 2
stage_layers = [4, 8, 12, 16, 20]
batch = 1
tokenizer = Tokenizer2D(quantization_levels=quantization)


max_len = 0

for i in range(10):
    mesh = meshes[i]
    vertices = mesh.x[:, 0:2]
    faces = mesh.faces
    tokens, info = tokenizer.tokenize(vertices, faces, verbose=False)
    token_sequence = len(tokens)
    if max_len < token_sequence:
        max_len = token_sequence


print(max_len)

embedder = Embedding(vocab_size=quantization+3,
                     d_model=d_model, max_len=max_len)
embeddings = embedder(torch.tensor(tokens))
embeddings.size()

mesh = meshes[1]
point_cloud = mesh.tri_coordinates[:, 0:2]

point_cloud.size()
point_cloud = point_cloud.unsqueeze(0)
point_cloud.size()
point_cloud_encoder = PerceiverPointEncoder(d_model, input_dim, n_latents)

latent_condition = point_cloud_encoder(point_cloud)
latent_condition.size()
tokens[0]

transformer = HourglassTransformer(
    d_model=d_model,
    n_heads=8,
    d_ff=4*d_model,
    dropout=0.1,
    shortening_method='attention',
    upsampling_method='attention',
    use_static_routing=True
)

embeddings = embedder(torch.tensor(tokens[:10]))
input = embeddings.unsqueeze(0)
input.size()

logig = transformer(x=input, latent_condition=latent_condition)
logig.size()
