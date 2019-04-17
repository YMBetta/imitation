import numpy as np
import random


class Sampler:
    def __init__(self, nenv, nsteps):
        self.indx = 0
        self.obs = np.loadtxt('env/obs.txt').reshape([nsteps+1, nenv, 2])[:-1, :, :].reshape([-1, 2])*10
        self.acs = np.loadtxt('env/obs.txt').reshape([nsteps+1, nenv, 2])[:-1, :, :].reshape([-1, 2])*100

    def next_sample(self):
        return self.obs, self.acs


