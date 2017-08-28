from vanilla_cyclegan.custom_dataset_data_loader import DatasetDataLoader
from vanilla_cyclegan.cyclegan import CycleGAN
from vanilla_cyclegan.params import *

dl = DatasetDataLoader(DATAROOT)

net = CycleGAN()

for epoch_idx in range(EPOCHS):
    for rollout_idx, data in dl.loader:
        rollout_img_x, rollout_img_y = dl.get_imgs(data)

        losses = net.train(rollout_img_x, rollout_img_y)

        print (losses)
