import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from fuel.datasets import H5PYDataset
from matplotlib.pyplot import imshow
from torch.utils.data import Dataset


class UnalignedMujocoDataset(Dataset):
    def __init__(self, dataroot):
        super(UnalignedMujocoDataset, self).__init__()
        file_path = os.path.join(dataroot, "data.h5")
        self.f = H5PYDataset(file_path, which_sets=('train',))  # TODO replace "train" with opt.phase

    # numpy image: H x W x C
    # torch image: C X H X W
    def _totensor(self, img):
        img = img.transpose((0, 3, 2, 1))
        return torch.from_numpy(img)

    def __getitem__(self, index):
        handle = self.f.open()
        data = self.f.get_data(handle, slice(index, index + 1))

        # items:
        # 0 - sin(joint 1 angle)
        # 1 - sin(joint 2 angle)
        # 2 - cos(joint 1 angle)
        # 3 - cos(joint 2 angle)
        # 4 - constant
        # 5 - constant
        # 6 - joint 1 velocity / angular momentum
        # 7 - joint 2 velocity / angular momentum
        # 8-10 distance fingertip to reward object

        relevant_items = [2, 3, 6, 7]  # both angles and velocities
        episode = {'state_joints': torch.from_numpy(data[2][0][:, relevant_items]),
                   'state_img': self._totensor(data[1][0]),
                   'action': torch.from_numpy(data[0][0]),
                   'state_next_sim_joints': torch.from_numpy(data[8][0][:, relevant_items]),
                   'state_next_sim_img': self._totensor(data[7][0]),
                   'state_next_real_joints': torch.from_numpy(data[4][0][:, relevant_items]),
                   'state_next_real_img': self._totensor(data[3][0])
                   }

        # print (episode["action"].size())
        # print (episode["state_joints"].size())
        #
        # print (episode["state_next_sim_joints"].size())
        # print (episode["state_next_real_joints"].size())
        #
        # print (episode["state_img"].size())
        # print (episode["state_next_sim_img"].size())
        # print (episode["state_next_real_img"].size())
        #
        self.f.close(handle)

        return episode

    def __len__(self):
        return self.f.num_examples


if __name__ == '__main__':
    ds = UnalignedMujocoDataset("../datasets/mujoco1/")

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=1)

    print(len(ds))
    for s in dl:
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
