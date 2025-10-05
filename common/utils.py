import os
import gc

import numpy as np
import subprocess
from time import localtime, strftime


def create_file(dir, pdb_name, gen, index, seed):
    fnew = f'{pdb_name}_s{seed}_gen{gen}_ind{index}.pdb'
    return os.path.join(dir, fnew)

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

def get_time():
    return strftime("%Y-%m-%d-%H:%M:%S", localtime())

def write_fasta(file, individual):
    with open(file, "w") as f:

        header_parts = [
            individual.name,
            f"generation={individual.generation}",
            f"index={individual.index}"
        ]
        
        for fitness_name, fitness_value in individual.fitnesses.items():
            header_parts.append(f"{fitness_name}={fitness_value}")
        
        header = ", ".join(header_parts)
        f.write(f">{header}\n{individual.sequence}")

def sample_indices(prob_matrix):
    row_sums = prob_tensor.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
        raise ValueError("Each row of prob_tensor must sum to 1. Got sums: {}".format(row_sums))

    sampled = torch.multinomial(prob_tensor, num_samples=1, replacement=True)

    assert sampled.size(0) == prob_tensor.size(0)
    return sampled.squeeze(1)
    