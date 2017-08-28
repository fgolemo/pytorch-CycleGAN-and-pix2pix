from torch import nn, optim

from vanilla_cyclegan.net_discriminator import NetDiscriminator
from vanilla_cyclegan.net_generator import NetGenerator
from .params import *


class CycleGAN():
    def __init__(self):
        self.gen_sim2real = NetGenerator()
        self.gen_real2sim = NetGenerator()
        self.disc_sim = NetDiscriminator()
        self.disc_real = NetDiscriminator()

        if CUDA:
            self.gen_sim2real = self.gen_sim2real.cuda()
            self.gen_real2sim = self.gen_real2sim.cuda()
            self.disc_sim = self.disc_sim.cuda()
            self.disc_real = self.disc_real.cuda()

        self.loss_function = nn.MSELoss()
        self.optimizer_gen_sim2real = optim.Adam(self.gen_sim2real.parameters())
        self.optimizer_gen_real2sim = optim.Adam(self.gen_real2sim.parameters())
        self.optimizer_disc_real = optim.Adam(self.disc_real.parameters())
        self.optimizer_disc_sim = optim.Adam(self.disc_sim.parameters())

    def get_discriminator_loss(self, data_sim, data_real, fake_sim, fake_real):
        # get loss for simulation discriminator
        # should identify sim data as 1
        disc_pred_sim = self.disc_sim.forward(data_sim)
        loss_disc_pred_sim = self.loss_function(disc_pred_sim - 1)
        # should identify fake sim data as 0
        disc_pred_fakesim = self.disc_sim.forward(fake_sim)
        loss_disc_pred_fakesim = self.loss_function(disc_pred_fakesim - 0)

        loss_disc_sim = loss_disc_pred_sim + loss_disc_pred_fakesim / 2

        # get loss for real discriminator
        # should identify real data as 1
        disc_pred_real = self.disc_real.forward(data_real)
        loss_disc_pred_real = self.loss_function(disc_pred_real - 1)
        # should identify fake real data as 0
        disc_pred_fakereal = self.disc_real.forward(fake_real)
        loss_disc_pred_fakereal = self.loss_function(disc_pred_fakereal - 0)

        loss_disc_real = loss_disc_pred_real + loss_disc_pred_fakereal / 2

        return loss_disc_sim, loss_disc_real, disc_pred_fakesim, disc_pred_fakereal

    def get_generator_loss(self, data_sim, data_real, disc_pred_fakesim, disc_pred_fakereal):
        # calculate the discriminator losses as how well they fool the discriminator
        loss_gen_fakesim = self.loss_function(disc_pred_fakesim - 1)
        loss_gen_fakereal = self.loss_function(disc_pred_fakereal - 1)

        # calculate the cyclic real and sim images
        cyclic_sim = self.gen_real2sim(self.gen_sim2real(data_sim))
        loss_cyclic_sim = self.loss_function(data_sim - cyclic_sim)

        cyclic_real = self.gen_sim2real(self.gen_real2sim(data_real))
        loss_cyclic_real = self.loss_function(data_real - cyclic_real)

        loss_gen_sim = loss_gen_fakesim + 10 * loss_cyclic_sim
        loss_gen_real = loss_gen_fakereal + 10 * loss_cyclic_real

        return loss_gen_sim, loss_gen_real

    def calculate_losses(self, data_sim, data_real):
        # convert sim 2 fake real
        fake_real = self.gen_sim2real.forward(data_sim)
        # convert real 2 fake sim
        fake_sim = self.gen_real2sim.forward(data_real)

        loss_disc_sim, \
        loss_disc_real, \
        disc_pred_fakesim, \
        disc_pred_fakereal = self.get_discriminator_loss(data_sim, data_real, fake_sim, fake_real)

        loss_gen_sim, \
        loss_gen_real = self.get_generator_loss(data_sim, data_real, disc_pred_fakesim, disc_pred_fakereal)

        loss_gen_sim.backward()
        loss_gen_real.backward()
        loss_disc_sim.backward()
        loss_disc_real.backward()

        return [loss_gen_sim, loss_gen_real, loss_disc_sim, loss_disc_real]

    def reset_optimizer(self):
        self.optimizer_gen_sim2real.zero_grad()
        self.optimizer_gen_real2sim.zero_grad()
        self.optimizer_disc_sim.zero_grad()
        self.optimizer_disc_real.zero_grad()

    def optimize_stuff(self):
        self.optimizer_gen_sim2real.step()
        self.optimizer_gen_real2sim.step()
        self.optimizer_disc_sim.step()
        self.optimizer_disc_real.step()

    def train(self, data_sim, data_real):
        self.reset_optimizer()
        self.calculate_losses(data_sim, data_real)
        self.optimize_stuff()
