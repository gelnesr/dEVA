import os
import tempfile
import numpy as np
from typing import Dict
from plip.structure.preparation import PDBComplex

from core.interfaces import BaseModel
from core.registry import register_model
from evolve.individual import Individual

@register_model("plip_score")
class ProtLigandScorer(BaseModel):
    def __init__(self, weights=None, metals=("ZN","MG","CA","MN","FE","CO","NI","CU","CD","NA","K","CL")):
        self.w = weights or {
            "hbond": 2.0, 
            "salt": 3.0, 
            "pi": 2.0, 
            "pication": 2.0,
            "halogen": 1.5, 
            "hydroph": 0.5, 
            "water": 0.5,
        }
        self.metals = set(metals)

    def setup(self, config: Dict, device: str='cpu') -> None:
        self.config = config
        self.device = device

        self.pdb = self.config.input.pdb
        self.pdb_name = os.path.split(self.pdb)[-1].split('.')[0]
        
        self.model_config = self.config.models.plip
        if self.model_config.weights is not None:
            self.w = self.model_config.weights
        if self.model_config.metals is not None:
            self.metals = set(metals)


        outputs = self.config.general.outputs
        self.output_tmp = os.path.join(outputs, "plip")
        os.makedirs(self.output_tmp, exist_ok=True)
        os.environ["TMPDIR"] = self.output_tmp
        tempfile.tempdir = self.output_tmp

    def score(self, individual: Individual):
        curr_pdb = individual.get_name()
    
        mol = PDBComplex()
        mol.load_pdb(curr_pdb)
        mol.analyze()

        keys = list(mol.interaction_sets.keys())
        keys = [k for k in keys if k.split(":")[0].strip() not in self.metals]
        if not keys:
            individual.add_fitness({'plip': 1.0})
        per = []
        out = {}

        for k in keys:
            it = mol.interaction_sets[k]
            hb = list(getattr(it, "hbonds_lig", [])) + list(getattr(it, "hbonds_pdon", []))
            salt = list(getattr(it, "saltbridge_lneg", [])) + list(getattr(it, "saltbridge_pneg", []))
            pi = list(getattr(it, "pistacking", []))
            pc = list(getattr(it, "pication", []))
            hal = list(getattr(it, "halogen_bonds", getattr(it, "halogen", [])))
            hyd = list(getattr(it, "hydrophobic_contacts", []))
            wat = list(getattr(it, "waterbridge", []))

            counts = {"hbond": len(hb), "salt": len(salt), "pi": len(pi), "pication": len(pc),
                      "halogen": len(hal), "hydroph": len(hyd), "water": len(wat)}

            s = sum(self.w[t] * counts[t] for t in counts)  # weighted sum of interactions
            s += 0.5 * sum(n > 0 for n in counts.values())  # diversity bonus
            tot = sum(counts.values())                  
            if tot < 3: s -= 2.0                            # penalty for too few contacts
            elif tot < 6: s -= 0.5

            out[k] = {"score": float(s), "counts": counts}
            per.append(float(s))

        individual.add_fitness({"plip": float(np.mean(per))})
