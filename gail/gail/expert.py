import numpy as np


class Sampler:
    def __init__(self, nenv, nsteps):
        self.indx = 0
        self.obs = np.loadtxt('env/obs.txt')
        self.acs = np.loadtxt('env/acs.txt')
        self.acs_min, self.acs_max = np.min(self.acs, axis=0), np.max(self.acs, axis=0)
        self.obs_min, self.obs_max = np.min(self.obs, axis=0), np.max(self.obs, axis=0)
        
    def next_sample(self):
#        return (self.obs-self.obs_min)/self.obs_max,  (self.acs-self.acs_min)/self.acs_max
        return self.obs, self.acs

