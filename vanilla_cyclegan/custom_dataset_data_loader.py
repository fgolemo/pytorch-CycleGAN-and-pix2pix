import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib.pyplot import imshow

from vanilla_cyclegan.params import *
from vanilla_cyclegan.unaligned_mujoco_dataset import UnalignedMujocoDataset


class DatasetDataLoader():
    def __init__(self, dataroot):
        self.source = UnalignedMujocoDataset("../datasets/mujoco1/")
        self.loader = torch.utils.data.DataLoader(
            self.source,
            batch_size=1,
            shuffle=True,
            num_workers=1)

    def get_imgs(self, data):
        # # we currently don't need the actual input image unless we wanna
        # # learn the simulator as well
        # current_img = data["state_img"][0]
        next_img_sim = data["state_next_sim_img"][0]
        next_img_real = data["state_next_real_img"][0]

        x = torch.autograd.Variable(next_img_sim, requires_grad=False)
        y = torch.autograd.Variable(next_img_real, requires_grad=False)
        if CUDA:
            return x.cuda(), y.cuda()

        return x, y

    def get_joints(self, data):
        next_joint_sim = data["state_next_sim_joints"][0]
        next_joint_real = data["state_next_real_joints"][0]

        x = torch.autograd.Variable(next_joint_sim, requires_grad=False)
        y = torch.autograd.Variable(next_joint_real, requires_grad=False)
        if CUDA:
            return x.cuda(), y.cuda()

        return x, y


if __name__ == '__main__':
    dl = DatasetDataLoader("../datasets/mujoco1/")
    for s in dl.loader:

        current_img = s["state_img"][0][0]
        next_img_sim = s["state_next_sim_img"][0][0]
        next_img_real = s["state_next_real_img"][0][0]

        imgs = [current_img, next_img_sim, next_img_real]
        for i in imgs:
            print("img sizes: ", i.size())

        combined = torchvision.utils.make_grid(imgs, 1)
        print(combined.size())
        imshow(np.swapaxes(combined.numpy(), 0, 2))
        plt.show()

        break
