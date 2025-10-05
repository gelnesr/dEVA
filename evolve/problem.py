import torch
import numpy as np

from common.utils import write_fasta
from evolve.individual import Individual

class Problem:
    def __init__(self, sampler=None, fitness_keys=None, seq_model="ligandmpnn"):
        self.design_sampler = sampler
        self.fitness_keys = fitness_keys
        self.seq_model = seq_model
        self.write_fasta = False

    def generate_individual(self, generation=0, index=0):
        individual = Individual(generation=generation, index=index)
        self.design_sampler.init_seq(individual)
        individual.add_header(individual.get_name())
        if self.write_fasta:
            # need to define fasta_file
            utils.write_fasta(fasta_file, individual)
        return individual
    
    def get_fitness_keys(self):
        """Get the fitness keys for this problem"""
        if self.fitness_keys is None:
            sample_individual = self.generate_individual(index=0)
            self.fitness_keys = list(sample_individual.fitnesses.keys())
        return self.fitness_keys
