

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import random
plt.style.use('ggplot')


__all__ = ['genetic_process', 'population'] # export only these classes



class gene:
	def __init__(self, allele):
		self.allele = allele # value of gene


class chromosome:
	def __init__(self, genes):
		self.genes_objects = np.array([gene(g) for g in genes]) # list of all genes objects
		self.genes = genes

	def fitness(self, model, data): # calculate the prediction score for each chromosome (features) in each generation
		model.fit(data["x_train"].iloc[:, self.genes], data["y_train"]) # fit the training data genes (features) into the model - only those genes that have True allele
		predictions = model.predict(data["x_test"].iloc[:, self.genes]) # predicting on testing data genes (features)
		return accuracy_score(data["y_test"], predictions) # use the accuracy as a fitness for each chromosome

	def __getitem__(self, locus):
		return self.genes_objects[locus]

	def __len__(self):
		return len(self.genes)


class population:
	def __init__(self, amount=200, features=30, chromosomes=None):
		self.amount = amount
		self.features = features
		self.pop = [] # list of all chromosomes (solutions)
		if not chromosomes: self.__init_pop()
		else: self.pop = [chromosome(c) for c in chromosomes]
	
	def __init_pop(self):
		for i in range(self.amount):
			c = np.ones(self.features, dtype=np.bool)
			c[:int(0.4*self.features)] = False # not all features are good! - binary representation
			np.random.shuffle(c) # shuffle all chromosomes which have the number of features (genes) in their structure in each population
			self.pop.append(chromosome(c))

	def fitness_score(self, model, data):
		scores = [] # prediction scores
		for chromosome in self.pop:
			scores.append(chromosome.fitness(model, data))
		scores, population = np.array(scores), np.array([c.genes for c in self.pop])
		indices = np.argsort(scores) # return the indices of sorted scores in ascending order - used in rank selection
		descending_scores = scores[indices][::-1] # sorted scores in descending order
		descending_population_of_scores = population[indices, :][::-1] # sorted population of chromosomes scores in descending order
		return list(descending_scores), list(descending_population_of_scores) # return descending order of population of none object genes chromosome and scores

	def __len__(self):
		return len(self.pop)


	def __getitem__(self, idx):
		return self.pop[idx]


class genetic_process:
	def __init__(self, generation, population, parents, model, selection_method, crossover_method, mutation_method, mutation_rate, **data):
		self.generation = generation
		self.population = population
		self.parents = parents
		self.model = model
		self.mutation_rate = mutation_rate
		self.data = data
		self.selection_method = selection_method
		self.crossover_method = crossover_method
		self.mutation_method = mutation_method
		self.population_after_fitness = []
		self.parents_population = []
		self.population_after_crossover = []
		self.best_chromosomes = []
		self.best_scores = []


	def run(self):
		for i in range(self.generation):
			print(f"🧬 Generation --- {i+1}")
			scores, self.population_after_fitness = self.population.fitness_score(self.model, self.data)
			print(f"\t▶  Best Score for Two Chromosomes --- {scores[:2]}\n") # the first two are the best ones!
			# =================== GA Operators ===================
			self.__selection() # select best fitness as parents
			self.__crossover() # parents mating pool
			self.__mutation() # mutating genes
			# ====================================================
			self.best_chromosomes.append(self.population_after_fitness[0]) # because population_after_fitness it's sorted in descending order, the first one is the best one in this generation
			self.best_scores.append(scores[0]) # sorted scores in descending order, so the first one is the higher one in this generation


	def __crossover(self):
		offspring = self.parents_population
		if self.crossover_method == "single_point":
			raise NotImplementedError # TODO
		elif self.crossover_method == "two_point":
			raise NotImplementedError # TODO
		elif self.crossover_method == "multi_point":
			for i in range(len(self.parents_population)): # half of it is the parents and the other half is the new offspring
				child = self.parents_population[i]
				child[3:7] = self.parents_population[(i+1)%len(self.parents_population)][3:7] # crossover takes palce between two parents
				offspring.append(child)
			self.population_after_crossover = offspring
		else:
			raise NotImplementedError


	def __mutation(self):
		offspring_after_mutation = []
		if self.mutation_method == "flipping":
			for i in range(len(self.population_after_crossover)):
				chromosome = self.population_after_crossover[i]
				for j in range(len(chromosome)):
					if random.random() < self.mutation_rate:
						chromosome[j] = not chromosome[j] # we can only mutate by changing the True to False or vise versa.
				offspring_after_mutation.append(chromosome)
			self.population = population(chromosomes=offspring_after_mutation)
		elif self.mutation_method == "reversing":
			raise NotImplementedError # TODO
		elif self.mutation_rate == "interchanging":
			raise NotImplementedError # TODO
		else:
			raise NotImplementedError

	def __selection(self):
		population_next_generation = []
		if self.selection_method == "roulette_wheel":
			fitness_population = sum(self.population.fitness_score(self.model, self.data)[0]) # sum of all scores (fitnesses)
			individual_expected_values = [(1/c.fitness(self.model, self.data))/fitness_population for c in self.population] # all chromosomes prob (exprected values) - we scaled up every chromosome fitness cause big roulette of the wheel belongs to minimum fitnesses
			cum_prob = [sum(individual_expected_values[:i+1]) for i in range(len(individual_expected_values))] # cumulative sum of chromosomes exprected values (prob)
			for i in range(self.parents):
				r = random.random()
				if cum_prob[i] >= r:
					population_next_generation.append(self.population[i].genes)
			self.parents_population = population_next_generation
		elif self.selection_method == "rank": # because the population after fitness is sorted in descending order we know that always the first n chromosomes of population are the best ones so it doesn't matter how many parents you're choosing to breed.
			for i in range(self.parents):
				population_next_generation.append(self.population_after_fitness[i])
			self.parents_population = population_next_generation
		elif self.selection_method == "tournament":
			raise NotImplementedError # TODO
		else:
			raise NotImplementedError

	def plot(self):			
		plt.plot(self.best_scores)
		plt.xlabel("Generation")
		plt.ylabel("Best Fitness")
		plt.savefig("fitness_generation.png")

	def save(self):
		print(f"▶ Saving Best chromosomes and best scores")
		np.save(f"best_chromo_in_{self.generation}_generations.npy", self.best_chromosomes)
		np.save(f"best_scores_in_{self.generation}_generations.npy", self.best_scores)
