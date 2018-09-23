import gym
import numpy as np
import math

# import numpy as np
# import gym
# import multiprocessing as mp
# import time

'''

Training Class
--------------
Solve MountainCar-v0 Gym Challenge with genetic evolutionary strategy

Inspired by:
------------
- https://arxiv.org/abs/1703.03864
- http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
- https://github.com/MorvanZhou/Evolutionary-Algorithm/blob/master/tutorial-contents/Using%20Neural%20Nets/Evolution%20Strategy%20with%20Neural%20Nets.py
------------

'''


class MountainCarSolver:

	def __init__(self, generations=2500):
		# Load MountainCar
		self.env = gym.make('MountainCar-v0')

		# Learning rate
		self.learningRate = 0.5

		# Exploration rate (mutations)
		self.explorationRate = 0.5

		# Generations
		# ---
		# Number of generations we will train on
		# Similar concept to `number of epochs`
		# At each generation a subset of the population mutates and
		# another subset dies and is replaced by children from other individuals
		self.generations = generations

		# Population size
		# ---
		# Number of individual in the population,
		# each generation has the same population size
		self.populationSize = 30

		# Number of kids per generation
		# ---
		# Kids replace individuals which did not survive the previous generation
		# Kids are copies of the best performing individuals with few mutations.
		# This allows to quickly explore the feature space, while 
		# encouraging best performing individual to last, and less performing to evolve
		self.kidsPerGeneration = 4

		# Create initial population
		# ---
		# Each individual is a randomly generated feed forward neural net
		# An individual role is to select the best `action` given an `observation`
		self.population = [Individual((
			len(self.env.observation_space.high), 
			30,
			20,
			self.env.action_space.n
		)) for i in range(self.populationSize)]

		# Start training
		self.run()


	# Run game
	def run(self):

		# Loop through generations
		for i in range(self.generations):

			# Loop through individuals
			for individual in self.population:
				# self.get_reward()

				# Get default state
				current_state = self.env.reset()
				episode_reward = 0.

				# An episode ends after 200 steps
				for step in range(200):
					# self.env.render()

					action = individual.forward(current_state)

					# Run step with chosen action
					# retrieve current action reward and new state
					new_state, reward, done, info = self.env.step(action)

					if new_state[0] > -0.1: 
						reward = 0.

					episode_reward += reward

					# Remember state
					current_state = new_state

					# break

				# Set individual's reward
				individual.reward(episode_reward)

				if episode_reward > -200:
					print episode_reward


			self.next_generation()


	def next_generation(self):
		def sort(e):
			return e._reward

		# sort population per reward
		self.population.sort(key=sort, reverse=True)



'''

Individual class
----------------
Simple `fast forward` neural net implementation

'''

class Individual:
	def __init__(self, shapes, layers=None):
		self.layers = []
		self._reward = 0

		# Create layers
		for i, shape in enumerate(shapes):
			if i < len(shapes) - 1:
				self.layers.append(self.layer(shapes[i], shapes[i + 1]))

		if layers:
			self.layers = layers


	# Network layer
	def layer(self, inputs, outputs):
		# Create layer inputs
		w = np.random.randn(inputs * outputs).astype(np.float32)
		w = w.reshape((inputs, outputs))

		# Create outputs
		b = np.random.randn(outputs).astype(np.float32) * .1
		b = b.reshape((1, outputs))

		return [w, b]


	# Forward function with discrete output
	def forward(self, x):

		# Forward X to layers
		for i, layer in enumerate(self.layers):
			if i < len(self.layers) - 1:
				x = np.tanh(x.dot(layer[0]) + layer[1])
			else:
				x = x.dot(layer[0]) + layer[1]

		# Return computed Y
		return np.argmax(x, axis=1)[0]

	# Evolve
	def backprop(self, learningRate):
		pass

	# Save reward
	def reward(self, reward=None):
		self._reward = reward



MountainCarSolver()

