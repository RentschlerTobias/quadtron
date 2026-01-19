
from trainer import Trainer
import torch
#
# path = '../data/quad_data.pt'
path = '../data/unstructured_quad_meshes_v2.pt'
#
stage_layers = [2, 1, 1, 1, 1]

trainer = Trainer(data_path=path, num_epochs=20, learning_rate=1e-3, batch_size=2, quantization=1024, d_model=256,
                  gradient_accumulation=4, window_size=None, stage_layers=stage_layers)

trainer.training()


#
# data = next(iter(trainer.val_loader))
# point_cloud = (data['point_cloud'])
# face_count = (data['face_count'])
# input = data['input_tokens']
#
# input.size()
# point_cloud.size()
# face_count.size()
# face_count.unsqueeze(dim=0)
# start = (input[:, 0:8])
# start.size()
# input.size()
# point_cloud[0].size()
# generated_tokens = trainer.model.generate(
#     point_cloud=point_cloud, face_count=face_count.unsqueeze(dim=0), start_tokens=start)
# mask = generated_tokens != quantization
# tokens = generated_tokens[mask]
#


def main():

    path = '../data/quad_data.pt'
    stage_layers = [2, 2, 2, 2, 2]

    trainer = Trainer(num_epochs=500, learning_rate=1e-3, batch_size=8, quantization=1024, d_model=256,
                      gradient_accumulation=4, window_size=8*150, stage_layers=stage_layers)
    trainer.training()


if __name__ == "__main__":
    main()
