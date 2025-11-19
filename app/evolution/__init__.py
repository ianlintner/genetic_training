"""
Genetic algorithm for evolutionary optimization of validator configuration.
"""
from typing import Dict, Any, List, Optional, Tuple
import random
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from deap import base, creator, tools, algorithms
from loguru import logger

from app.validators import HybridValidator
from app.config import get_config
from app.training import TrainingPipeline


@dataclass
class Chromosome:
    """Represents a configuration chromosome for evolution."""
    heuristic_weights: Dict[str, float]
    judge_weight: float
    heuristics_weight: float
    thresholds: Dict[str, float]
    revision_depth: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "heuristic_weights": self.heuristic_weights,
            "judge_weight": self.judge_weight,
            "heuristics_weight": self.heuristics_weight,
            "thresholds": self.thresholds,
            "revision_depth": self.revision_depth
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chromosome":
        """Create from dictionary."""
        return cls(
            heuristic_weights=data["heuristic_weights"],
            judge_weight=data["judge_weight"],
            heuristics_weight=data["heuristics_weight"],
            thresholds=data["thresholds"],
            revision_depth=data["revision_depth"]
        )
    
    @classmethod
    def random(cls, config: Dict[str, Any]) -> "Chromosome":
        """Create a random chromosome."""
        # Random heuristic weights
        heuristic_names = [
            "token_length", "repetition", "structure",
            "harmful_content", "contradiction", "hallucination"
        ]
        heuristic_weights = {
            name: random.uniform(0.05, 0.30)
            for name in heuristic_names
        }
        
        # Normalize
        total = sum(heuristic_weights.values())
        heuristic_weights = {k: v / total for k, v in heuristic_weights.items()}
        
        # Random meta weights
        heuristics_weight = random.uniform(0.2, 0.8)
        judge_weight = 1.0 - heuristics_weight
        
        # Random thresholds
        thresholds = {
            "validation": random.uniform(0.5, 0.9),
            "revision": random.uniform(0.6, 0.95)
        }
        
        # Random revision depth
        revision_depth = random.randint(1, 5)
        
        return cls(
            heuristic_weights=heuristic_weights,
            judge_weight=judge_weight,
            heuristics_weight=heuristics_weight,
            thresholds=thresholds,
            revision_depth=revision_depth
        )


class GeneticOptimizer:
    """Genetic algorithm for optimizing validator configuration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize genetic optimizer.
        
        Args:
            config: Evolution configuration
        """
        if config is None:
            app_config = get_config()
            config = app_config.evolution
        
        self.config = config
        self.population_size = config.get("population_size", 20)
        self.generations = config.get("generations", 50)
        self.mutation_rate = config.get("mutation_rate", 0.2)
        self.crossover_rate = config.get("crossover_rate", 0.7)
        self.elitism = config.get("elitism", 0.1)
        
        self.training_pipeline = TrainingPipeline()
        self._setup_deap()
    
    def _setup_deap(self):
        """Setup DEAP framework."""
        # Create fitness and individual classes
        try:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        except AttributeError:
            pass
        try:
            creator.create("Individual", list, fitness=creator.FitnessMax, chromosome=None)
        except AttributeError:
            pass
        
        self.toolbox = base.Toolbox()
        
        # Register creation functions
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self._evaluate_fitness)
    
    def _create_individual(self):
        """Create a random individual."""
        chromosome = Chromosome.random(self.config)
        individual = creator.Individual([chromosome])
        individual.chromosome = chromosome
        return individual
    
    def _crossover(self, ind1, ind2) -> Tuple:
        """Crossover two individuals."""
        if random.random() > self.crossover_rate:
            return ind1, ind2
        
        c1 = ind1.chromosome
        c2 = ind2.chromosome
        
        # Crossover heuristic weights
        child1_hw = {}
        child2_hw = {}
        for key in c1.heuristic_weights:
            if random.random() < 0.5:
                child1_hw[key] = c1.heuristic_weights[key]
                child2_hw[key] = c2.heuristic_weights[key]
            else:
                child1_hw[key] = c2.heuristic_weights[key]
                child2_hw[key] = c1.heuristic_weights[key]
        
        # Normalize
        total1 = sum(child1_hw.values())
        total2 = sum(child2_hw.values())
        child1_hw = {k: v / total1 for k, v in child1_hw.items()}
        child2_hw = {k: v / total2 for k, v in child2_hw.items()}
        
        # Crossover other parameters
        alpha = random.random()
        
        new_c1 = Chromosome(
            heuristic_weights=child1_hw,
            judge_weight=alpha * c1.judge_weight + (1 - alpha) * c2.judge_weight,
            heuristics_weight=alpha * c1.heuristics_weight + (1 - alpha) * c2.heuristics_weight,
            thresholds={
                k: alpha * c1.thresholds[k] + (1 - alpha) * c2.thresholds[k]
                for k in c1.thresholds
            },
            revision_depth=c1.revision_depth if random.random() < 0.5 else c2.revision_depth
        )
        
        new_c2 = Chromosome(
            heuristic_weights=child2_hw,
            judge_weight=(1 - alpha) * c1.judge_weight + alpha * c2.judge_weight,
            heuristics_weight=(1 - alpha) * c1.heuristics_weight + alpha * c2.heuristics_weight,
            thresholds={
                k: (1 - alpha) * c1.thresholds[k] + alpha * c2.thresholds[k]
                for k in c1.thresholds
            },
            revision_depth=c2.revision_depth if random.random() < 0.5 else c1.revision_depth
        )
        
        ind1[0] = new_c1
        ind1.chromosome = new_c1
        ind2[0] = new_c2
        ind2.chromosome = new_c2
        
        return ind1, ind2
    
    def _mutate(self, individual) -> Tuple:
        """Mutate an individual."""
        if random.random() > self.mutation_rate:
            return (individual,)
        
        chromosome = deepcopy(individual.chromosome)
        mutation_ranges = self.config.get("mutation_ranges", {})
        
        # Mutate heuristic weights
        if random.random() < 0.5:
            key = random.choice(list(chromosome.heuristic_weights.keys()))
            delta = random.gauss(0, mutation_ranges.get("heuristic_weights", 0.1))
            chromosome.heuristic_weights[key] = max(0.01, chromosome.heuristic_weights[key] + delta)
            
            # Normalize
            total = sum(chromosome.heuristic_weights.values())
            chromosome.heuristic_weights = {k: v / total for k, v in chromosome.heuristic_weights.items()}
        
        # Mutate meta weights
        if random.random() < 0.3:
            delta = random.gauss(0, mutation_ranges.get("judge_weight", 0.1))
            chromosome.heuristics_weight = max(0.1, min(0.9, chromosome.heuristics_weight + delta))
            chromosome.judge_weight = 1.0 - chromosome.heuristics_weight
        
        # Mutate thresholds
        if random.random() < 0.3:
            for key in chromosome.thresholds:
                delta = random.gauss(0, mutation_ranges.get("thresholds", 0.05))
                chromosome.thresholds[key] = max(0.3, min(0.95, chromosome.thresholds[key] + delta))
        
        # Mutate revision depth
        if random.random() < 0.2:
            delta = random.choice([-1, 1])
            chromosome.revision_depth = max(1, min(5, chromosome.revision_depth + delta))
        
        individual[0] = chromosome
        individual.chromosome = chromosome
        
        return (individual,)
    
    def _evaluate_fitness(self, individual) -> Tuple[float]:
        """Evaluate fitness of an individual."""
        chromosome = individual.chromosome
        
        try:
            # Create validator with chromosome configuration
            validator = HybridValidator()
            validator.weights = {
                "heuristics": chromosome.heuristics_weight,
                "judge": chromosome.judge_weight
            }
            
            # Update heuristic weights
            for heuristic in validator.heuristics:
                name = heuristic.get_name()
                if name in chromosome.heuristic_weights:
                    heuristic.weight = chromosome.heuristic_weights[name]
            
            # Use training pipeline to evaluate
            self.training_pipeline.validator = validator
            
            # Load a small test dataset
            test_data = self.training_pipeline._create_mock_dataset(50)
            metrics = self.training_pipeline.evaluate_on_dataset(test_data)
            
            # Fitness is F1 score (or negative error rate)
            fitness_metric = self.config.get("fitness", {}).get("metric", "f1")
            minimize = self.config.get("fitness", {}).get("minimize", False)
            
            if fitness_metric == "error_rate":
                fitness = 1.0 - metrics["accuracy"]
                if not minimize:
                    fitness = 1.0 - fitness
            else:
                fitness = metrics.get(fitness_metric, metrics["f1"])
                if minimize:
                    fitness = 1.0 - fitness
            
            return (fitness,)
            
        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            return (0.0,)
    
    def evolve(self) -> Dict[str, Any]:
        """
        Run genetic algorithm evolution.
        
        Returns:
            Evolution results with best configuration
        """
        logger.info(f"Starting evolution: {self.population_size} individuals, {self.generations} generations")
        
        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        logger.info("Evaluating initial population...")
        fitnesses = map(self.toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Track statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields
        
        # Evolution loop
        for gen in range(self.generations):
            logger.info(f"Generation {gen + 1}/{self.generations}")
            
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Elitism: keep best individuals
            elite_count = int(self.elitism * self.population_size)
            if elite_count > 0:
                elite = tools.selBest(population, elite_count)
                offspring[- elite_count:] = elite
            
            # Replace population
            population[:] = offspring
            
            # Record statistics
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            logger.info(f"  Fitness - avg: {record['avg']:.4f}, max: {record['max']:.4f}")
        
        # Get best individual
        best_ind = tools.selBest(population, 1)[0]
        best_chromosome = best_ind.chromosome
        
        logger.info(f"Evolution complete!")
        logger.info(f"Best fitness: {best_ind.fitness.values[0]:.4f}")
        logger.info(f"Best configuration: {best_chromosome.to_dict()}")
        
        return {
            "best_configuration": best_chromosome.to_dict(),
            "best_fitness": best_ind.fitness.values[0],
            "logbook": [dict(zip(logbook.header, entry)) for entry in logbook],
            "final_population": [ind.chromosome.to_dict() for ind in population]
        }
