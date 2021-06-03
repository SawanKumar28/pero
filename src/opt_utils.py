import numpy as np
from tqdm import tqdm
from src.misc_utils import PermutationHandler
import pdb

class SInstance():
    def __init__(self, permutation):
        self.permutation = permutation

class GA():
    def get_default_config():
        config = {
            'population_size': 100,
            'keep_size': 25,
            'n_examples': 10,
            'permutation_size': 10,
            'random_seed': 0,
            'mutation_p': 0.1,
            'elite_ratio': 0.1,
            'do_crossover': True,
            'do_mutation': True,
            'selection': 'top'
        }
        return config

    def __init__(self, config, callbacks):
        print ('GA', config)
        self.config = config
        self.callbacks = callbacks

        self.elite_size = int(config['population_size'] * config['elite_ratio'])

        self.rng = np.random.RandomState(seed=config['random_seed'])
        self.PH = PermutationHandler(config['n_examples'], k=config['permutation_size'], random_seed=config['random_seed'])

        self.current_fitness = None
        self.initialize()

    def initialize(self):
        config = self.config
        self.population = []
        for _ in range(config['population_size']):
            perm = self.PH.generate_random_permutation()
            entry = SInstance(perm)
            self.population.append(entry)

    def mutate(self, item):
        config = self.config
        perm = item.permutation
        rand = self.rng.random(size=len(perm))
        for idx in range(len(perm)):
            if rand[idx] < config['mutation_p']:
                perm = self.PH.mutate(perm, pos=idx)
        return SInstance(perm)
            
    def crossover(self, item1, item2):
        perm1, perm2 = item1.permutation, item2.permutation
        cperm1, cperm2, cperm3, cperm4 = self.PH.crossover(perm1, perm2)

        return SInstance(cperm1), SInstance(cperm2), SInstance(cperm3), SInstance(cperm4)

    def compute_fitness(self):
        config = self.config
        fitness = np.zeros(len(self.population))
        if self.current_fitness is not None:
            fitness[:self.elite_size] = self.current_fitness[:self.elite_size]
            start_idx = self.elite_size
        else:
            start_idx = 0
        start_idx = 0

        for item_idx in range(start_idx, len(self.population)):
            fitness[item_idx] = self.callbacks['fitness_fn'](self.population[item_idx].permutation)
        return fitness

    def select(self):
        config = self.config
        #remove duplicates
        instances = []
        unique_set = set()
        for instance in self.population:
            if instance.permutation in unique_set: continue
            unique_set.add(instance.permutation)
            instances.append(instance)
        if len(instances) < len(self.population):
            print ("Removing duplicates {} -> {}".format(len(self.population), len(instances)))
        self.population = instances

        fitness = self.compute_fitness()
        so = np.argsort(fitness)
        pool_indices = so[:config['keep_size']]
        if config['selection'] == 'sample':
            sel_indices = self.rng.choice(pool_indices, size=len(pool_indices), p=self.keep_prob)
            #sort for book-keeping
            sel_fitness = fitness[sel_indices]
            so = np.argsort(sel_fitness)
            sel_indices = sel_indices[so]
            self.current_fitness = fitness[sel_indices]
            self.population = [self.population[idx] for idx in sel_indices]
        elif config['selection'] == 'top':
            self.current_fitness = fitness[pool_indices]
            self.population = [self.population[idx] for idx in pool_indices]
        else:
            print ("Selection strategy not implemented")

    def step(self):
        config = self.config
        self.select()
        if self.elite_size > 0:
            elite_population = self.population[:self.elite_size]
        else:
            elite_population = []
        
        n_new = config['population_size'] - len(elite_population)
        
        if config['do_crossover']:
            new_individuals = []
            indices = self.rng.choice(len(self.population), size=n_new*2)
            for idx in range(0, n_new, 2):
                new_individuals.extend(self.crossover(self.population[indices[idx]], self.population[indices[idx+1]]))
            mutation_input = new_individuals
        else:
            mutation_input = self.population

        if config['do_mutation']:
            indices = self.rng.choice(len(mutation_input), size=n_new)
            new_individuals = [self.mutate(mutation_input[idx]) for idx in indices]
        else:
            new_individuals = mutation_input[:n_new]
        
        self.population = elite_population + new_individuals

    def get_best_entry(self):
        return self.population[0]

optimizers = {
    'GA': GA,
}
