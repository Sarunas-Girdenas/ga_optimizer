from chromosome import Chromosome
from dataclasses import dataclass
from dask.distributed import Client
from typing import List
import pandas as pd
from numpy import argsort, array, dot, random

@dataclass
class GeneticAlgorithm():
    """Run genetic algorithm
    """

    data: pd.core.frame.DataFrame
    num_populations: int = 2
    yield_column: str = 'yield'
    
    def _generate_random_intervals_for_constraints(self):
        """Given data, generate random numbers that match
        the constraints
        Inputs:
        =======
        data (dataframe): used to generate constraints
        
        Outputs:
        ========
        constrained_initial_weights (dict): dict of lists of weights
        """
        
        # constraints - HY - no more than 0.01 and IG no more than 0.01
        
        ig_size = self.data.query("bond_type == 'IG'").shape[0]
        hy_size = self.data.query("bond_type == 'HY'").shape[0]
        
        # generate random guess
        d = hy_size + 2 * ig_size
        hy_ = random.uniform(0, hy_size/d, hy_size)
        ig_ = random.uniform(hy_size/d, 0.43, ig_size)
        
        # normalise
        hy_ = (hy_/sum(hy_) * hy_size) / 100
        ig_ = (ig_/sum(ig_) * ig_size) / 100
        
        # create generators
        constrained_initial_weights = {}
        constrained_initial_weights['hy'] = iter(hy_)
        constrained_initial_weights['ig'] = iter(ig_)

        return constrained_initial_weights

    def generate_initial_population(self):
        """Generate initial population
        Inputs:
        =======
        size_of_initial_population (int)

        Outputs:
        ========
        initial_population (list)
        """
        
        # generate random weights
        random_weights = self._generate_random_intervals_for_constraints()

        initial_population = []

        for row in self.data.iterrows():
            
            if row[1]['rating'] >= 11:
                max_weight = 0.01
                chromosome = Chromosome({'rating': row[1]['rating'],
                                        'yield': row[1]['yield'],
                                        'min_weight': 0,
                                        'max_weight': max_weight,
                                        'weight': round(next(random_weights['hy']), 3)})

            if row[1]['rating'] <= 10:
                max_weight = 0.02
                chromosome = Chromosome({'rating': row[1]['rating'],
                                        'yield': row[1]['yield'],
                                        'min_weight': 0,
                                        'max_weight': max_weight,
                                        'weight': round(next(random_weights['ig']), 3)})

            initial_population.append(chromosome)

        return initial_population
    
    @staticmethod
    def evaluate_fitness(inputs):
        """Given the chromosome and the data, evaluate the fitness
        Inputs:
        =======
        inputs (tuple): self.data (pd.core.frame.DataFrame),
            population (list of Chromosome instances): population to evaluate,
            yield_column (str): column in the data that represents yield

        Outputs:
        ========
        fitness (float): how fit is the chromosome
        """

        data, population, yield_column = inputs[0], inputs[1], inputs[2]
        weights = array([population[i].weight for i in range(len(population))])

        return dot(weights, data[yield_column].values)

    def rank_populations(self, top: int=0.5):
        """Given all the populations, rank them according to the
        the given fitness function
        Inputs:
        =======
        populations (List): populations to evaluate
        top (int): percentage of top populations to return

        Outputs:
        ========
        best_populations (List): top populations
        """

        client = Client()
        
        client_input = [(self.data, p, self.yield_column) for p in self.populations]
        
        futures = client.map(GeneticAlgorithm.evaluate_fitness, client_input)
        ranking = client.gather(futures)
        
        client.close()
        
        # return top performing populations
        top_n = int(top*len(self.populations))

        return [self.populations[i] for i in argsort(ranking)[-top_n:]]
    
    @staticmethod
    def _compute_parents_indices(population_size: int) -> List:
        """Given population size, calculate indices for parents
        Inputs:
        =======
        population_size (int): size of the population
        
        Outputs:
        ========
        parents_index (list): [(0, 1), (2, 3), ...]
        """
        parents_index = []

        for idx, i in enumerate(range(population_size)):
            if i == 0:
                id_ = (0, 1)
                parents_index.append(id_)
            else:
                id_ = (i+idx, i+idx+1)
                parents_index.append(id_)
        return parents_index

    def crossover(self,):
        """crossover is made as follows:
        a. Select 2 parents: G1, G2
        b. generate uniformly distributed random number gamma from [-alpha, 1 + alpha], where alpha = 0.5
        c. generate an offspring as follows: G = gamma * G1 + (1 - gamma) * G2
        """
        
        raise NotImplementedError
    
    def run(self):
        """Run Genetic Algorithm
        """
        
        # 1. Generate initial populations
        self.populations = [self.generate_initial_population() for _ in range(self.num_populations)]
        
        # 2. Evaluate fitness
        self.best_populations = self.rank_populations()
        
        # 3. Crossover
        self.crossover()

        return None