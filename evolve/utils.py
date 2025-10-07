import torch
import random
import logging
import numpy as np
from copy import deepcopy

from evolve.population import Population
from evolve.individual import Individual

class NSGA2Utils:
    def __init__(
        self,
        problem,
        num_individuals=100,
        num_tour_particips=10,
        tournament_prob=0.4,
        crossover_param=2,
        num_mutations=4,
        seed=0,
    ):
        self.problem = problem
        self.num_individuals = num_individuals
        self.num_tour_particips = num_tour_particips
        self.tournament_prob = tournament_prob
        self.crossover_param = crossover_param
        self.seed = seed
        self.num_mutations = num_mutations
        self.fixed_idx = None

    def create_initial_population(self, start_index=0):
        population = Population()

        individuals = []
        for i in range(start_index, self.num_individuals, 1):
            print(f'Creating {i}')
            individual = self.problem.generate_individual(generation=0, index=i)
            individuals.append(individual)
    
        population.extend(individuals)
        return population

    def fast_nondominated_sort(self, population):
        population.fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            population.fronts.append(temp)

    def calculate_crowding_distance(self, front):
        if len(front) <= 0:
            return
            
        solutions_num = len(front)
        for individual in front:
            individual.crowding_distance = 0

        fitness_keys = list(front[0].fitnesses.keys())
        
        for m in fitness_keys:
            # Sort by current fitness dimension
            front.sort(key=lambda individual: individual.fitnesses[m])
            
            front[0].crowding_distance = 10**9
            front[solutions_num - 1].crowding_distance = 10**9
            
            if solutions_num > 2:
                m_values = np.array([individual.fitnesses[m] for individual in front])
                scale = m_values.max() - m_values.min()
                if scale == 0:
                    scale = 1
                
                diffs = m_values[2:] - m_values[:-2]
                crowding_dists = diffs / scale
                
                for i, dist in enumerate(crowding_dists, start=1):
                    front[i].crowding_distance += dist

    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or (
            (individual.rank == other_individual.rank)
            and (individual.crowding_distance > other_individual.crowding_distance)
        ):
            return 1
        else:
            return -1

    def create_children(self, population, generation=0, sampler=None):
        children = []

        idx = 0
        if self.fixed_idx is None:
            self.fixed_idx = sampler.get_fixed_residues()

        while len(children) < len(population):
            parent1, parent2 = self.__tournament(population)

            while parent1.name == parent2.name:
                _, parent2 = self.__tournament(population)

            child1, child2 = self.__crossover(parent1, parent2, generation=generation, index=idx)

            # in this step new fitnesses are calculated
            self.__mutate(child1, sampler)
            self.__mutate(child2, sampler)
            children.append(child1)
            children.append(child2)
            
            idx += 2

        return children

    def __crossover(self, individual1, individual2, generation, index, var_idx=None):

        child1 = Individual(index=index, generation=generation)
        child2 = Individual(index=index+1, generation=generation)
        
        # Copy basic attributes more efficiently
        for child, parent in [(child1, individual1), (child2, individual2)]:
            child.name = parent.name
            child.sequence_ = parent.sequence_.clone()

        device = child1.sequence_.device
        L = len(individual1.sequence)

        k1, k2 = sorted([random.randint(0, L), random.randint(0, L)])
        seg_mask = torch.zeros(L, dtype=torch.bool, device=device).squeeze()
        seg_mask[k1:k2] = True

        self.fixed_idx = self.fixed_idx.to(device)
        # swap only where segment AND mutable
        swap_mask = seg_mask & self.fixed_idx
        swap_mask = swap_mask.unsqueeze(0) 

        child1.sequence_[swap_mask] = individual2.sequence_[swap_mask]
        child2.sequence_[swap_mask] = individual1.sequence_[swap_mask]
        
        return child1, child2

    def __mutate(self, child, sampler):
        sampler.step(individual=child, num_mutations=self.num_mutations)
        
        fitness_str = " ".join(f"{k}={v}" for k, v in child.fitnesses.items())
        
        print(f"generation {child.generation}, index {child.index}, {fitness_str}")
        print(f"final sequence: {child.sequence}\n")
    
    def __tournament(self, population):
        participants = random.sample(population.population, self.num_tour_particips)

        best = None
        second_best = None
        for participant in participants:
            if best is None or (
                self.crowding_operator(participant, best) == 1
                and self.__choose_with_prob(self.tournament_prob)
            ):
                second_best = best
                best = participant
            elif second_best is None or (
                self.crowding_operator(participant, second_best) == 1
                and self.__choose_with_prob(self.tournament_prob)
            ):
                second_best = participant

        return best, second_best

    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False

