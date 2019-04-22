We have 10 expert episodes, each with same length 100 steps. 
	acs.txt:  expert actions  file. Each row is an action, 1000 rows in total.  And every 100 continuted rows is all actions of an episode.
	obs.txt:  expert observations file. Each row is an observation, 1000 rows in total. And every 100 continuted rows is all observations of an episode.

How to load expert episodes:
	import numpy as np
	obs = np.loadtxt('obs.txt')
	acs = np.loadtxt('acs.txt')
	
How to simulate episodes:
	from env import Env2d
	env = Env2d()
	for _ in range(10):  # generate 10 episodes each with fixed 100steps randomly.
		env.reset()  # We must reset env manully when reach the maximum_steps.
		for i in range(100):
			action = np.random.random(2)
			obs, done = env.step(action)




	
