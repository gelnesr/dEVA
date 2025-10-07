import gc 
import os
import time
import torch
import subprocess
import numpy as np

from evolve.individual import Individual

class Sampler(object):
    def __init__(self, models):
        super(Sampler, self).__init__()
        self.rem_models = models
        self.seq_model = self.rem_models.pop('seq_model')
        self.fixed_residues = None
        if hasattr(self.seq_model, 'init_seq') and callable(getattr(self.seq_model, 'init_seq')):
            pass
        else:
            raise ValueError("Sequence design model does not have init_seq function")
        pass
    
    def init_seq(self, individual: Individual):
        self.get_fixed_residues()
        
        self.seq_model.init_seq(individual)
        for k, m in self.rem_models.items():
            m.score(individual)
    
    def step(self, individual, num_mutations=1):
        self.seq_model.score(individual, num_mutations=num_mutations)
        for k, m in self.rem_models.items():
            m.score(individual)

    def get_fixed_residues(self):
        self.fixed_residues = self.seq_model.fixed_resis()
        return self.fixed_residues
