import re
import os
import glob
import pickle
import logging
from tqdm import tqdm

from evolve.utils import NSGA2Utils
from evolve.population import Population

logger = logging.getLogger("evolution")
logger.setLevel(logging.DEBUG)

class Evolution:
    def __init__(
        self,
        problem,
        num_generations=10,
        num_individuals=10,
        num_tour_particips=2,
        tournament_prob=0.9,
        crossover_param=2,
        num_mutations=4,
        sampler=None,
        seed=0,
        checkpoint_freq=1,
        checkpoint_dir=None,  # Checkpoint directory for auto-loading
    ):
        self.utils = NSGA2Utils(
            problem,
            num_individuals,
            num_tour_particips,
            tournament_prob,
            crossover_param,
            num_mutations,
            seed=0,
        )
        self.population = None
        self.num_generations = num_generations
        self.on_generation_finished = []
        self.num_individuals = num_individuals
        self.sampler = sampler
        self.children = None
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq

        logger.info(f"Checkpoint directory set to: {self.checkpoint_dir}")

    def reinitialize(self):
        self.utils.reinitialize(self.sampler)

    def find_latest_checkpoint(self):   
        """Find the latest checkpoint file in the checkpoint directory"""
        
        # Look for checkpoint files with pattern pareto_front_gen*.pkl
        checkpoint_pattern = os.path.join(self.checkpoint_dir, "pareto_front_gen*.pkl")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            logger.info("No checkpoint files found")
            return None, 0
        
        # Extract generation numbers and find the latest
        latest_gen = 0
        latest_file = None
        
        for file_path in checkpoint_files:
            filename = os.path.basename(file_path)
            match = re.search(r'gen(\d+)', filename)
            if match:
                gen_num = int(match.group(1))
                logger.info(f"Found checkpoint file {filename} with generation {gen_num}")
                if gen_num > latest_gen:
                    latest_gen = gen_num
                    latest_file = file_path
        
        logger.info(f"Latest checkpoint: {latest_file} (generation {latest_gen})")
        return latest_file, latest_gen

    def load_checkpoint(self, checkpoint_file):
        self.reinitialize()
        """Load population from checkpoint file"""
        with open(checkpoint_file, "rb") as f:
            pareto_front = pickle.load(f)

        self.population = Population()
        self.population.extend(pareto_front)

        while len(self.population) < self.num_individuals:
            new_individuals = self.utils.create_initial_population(start_index=len(self.population))
            for ind in new_individuals:
                if len(self.population) >= self.num_individuals:
                    break
                if not any(existing.sequence == ind.sequence for existing in self.population):
                    self.population.append(ind)
            
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
            
        logger.info(f"Loaded checkpoint from {checkpoint_file} with {len(self.population)} individuals")
        return True

    def evolve(self):
        self.reinitialize()
        # Auto-detect latest checkpoint or start fresh
        latest_checkpoint, latest_gen = self.find_latest_checkpoint()
        if latest_checkpoint:
            logger.info(f"Auto-detected latest checkpoint: {latest_checkpoint} (generation {latest_gen})")
            if self.load_checkpoint(latest_checkpoint):
                start_gen = latest_gen + 1
                logger.info(f"Resuming from generation {start_gen}")
            else:
                logger.warning(f"Failed to load auto-detected checkpoint, starting fresh")
                start_gen = 1
                self.population = self.utils.create_initial_population()
                logger.debug("Initial population created")
                self.utils.fast_nondominated_sort(self.population)
                for front in self.population.fronts:
                    self.utils.calculate_crowding_distance(front)
                logger.debug("Initial population sorted")
        else:
            logger.info("No checkpoint files found, starting fresh")
            start_gen = 1
            self.population = self.utils.create_initial_population()
            logger.debug("Initial population created")
            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            logger.debug("Initial population sorted")
    
        statistics = []
        first_fronts = []
        for gen in tqdm(range(start_gen, self.num_generations+1)):
            self.children = self.utils.create_children(self.population, 
                                                       sampler=self.sampler, 
                                                       generation=gen)
            logger.debug(f"Children of Generation {gen} created")
            self.population.extend(self.children)
            
            fitness_keys = []
            if self.population:
                fitness_keys = list(self.population.population[0].fitnesses.keys())
            
            curr_gen = {"sequences": []}
            for key in fitness_keys:
                curr_gen[key] = []
            
            for i, ind in enumerate(self.population):
                self.population.population[i].index = i
                curr_gen["sequences"].append(ind.sequence)
                for key in fitness_keys:
                    curr_gen[key].append(ind.fitnesses.get(key, 0.0))
            statistics.append(curr_gen)
            self.utils.fast_nondominated_sort(self.population)
            new_population = Population()
            front_num = 0
            
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_individuals:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
        
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            new_population.extend(self.population.fronts[front_num][0 : self.num_individuals - len(new_population)])
            self.population = new_population
            self.utils.fast_nondominated_sort(self.population)

            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            first_fronts.append(self.population.fronts[0])

            # === Pareto front checkpointing every 5 generations ===
            if gen % self.checkpoint_freq == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"pareto_front_gen{gen}.pkl")
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(self.population.fronts[0], f)

        evo_out = {'best': self.population.fronts[0], 'statistics': statistics, 'fronts': first_fronts}
        return evo_out
