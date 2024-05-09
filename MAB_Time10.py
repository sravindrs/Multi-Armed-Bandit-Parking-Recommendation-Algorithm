import math
import numpy as np
import pandas as pd
from statistics import mean
from statistics import variance

# initialize random number generator
rng = np.random.default_rng(0)

# numbers of arms (parking lots), rounds 
N = 10
T = 1000

# number of times to run each randomized algorithm, and lists for storing total payouts from each run
K = 100
simple_payouts = []
exp3_payouts = []
ts_payouts = []

# lists of 1000 payouts (losses) for each arm
payouts = [ [] for i in range(N) ]

# read in payouts from csv file
data = pd.read_csv("/Users/andrewplattel/Downloads/STOR 515/Final Project/input_data.csv")
payouts[0] = data['Rams 4-5']
payouts[1] = data['Morehead 4-5']
payouts[2] = data['Raleigh Road Lot 4-5']
payouts[3] = data['Raleigh Rd 4-5']
payouts[4] = data['Country Club 4-5']
payouts[5] = data['Raleigh St 4-5']
payouts[6] = data['Wilson 4-5']
payouts[7] = data['UL 4-5']
payouts[8] = data['FedEx 4-5']
payouts[9] = data['Davis 4-5']

labels = ['Rams 4-5', 'Morehead 4-5', 'Raleigh Road Lot 4-5', 'Raleigh Rd 4-5', 'Country Club 4-5', 'Raleigh St 4-5', 'Wilson 4-5', 'UL 4-5', 'FedEx 4-5', 'Davis 4-5']


###############################################################

# Simple MAB algorithm with MWU as chosen full-information algorithm A

# probability of exploring
delta = 0.4

# pick a practical value for epsilon for MWU
epsilon = np.sqrt(np.log(N) / (T))

# run algorithm K times
for k in range(K):

	# keep track of cumulative payout for Simple MAB
	simple_payout = 0

	# initial vector of arm selection probabilities
	x = [(1/N) for i in range(N)]

	# vector of weights for MWU
	weights = [1 for i in range(N)]

	# algorithm
	for t in range(T):

		# choose whether to explore or exploit
		explore_or_exploit = rng.binomial(1, delta)
		
		# exploration
		if explore_or_exploit == 1:

			# choose arm to pull uniformly at random
			chosen_arm = rng.integers(low=0, high=N)

			# observe loss--will be either 1 (no payout) or 0 (payout)--and update cumulative payout
			loss = 1-payouts[chosen_arm][t]
			simple_payout += payouts[chosen_arm][t]

			# update weight of chosen arm
			weights[chosen_arm] *= (1-epsilon*loss)

			# update vector of selection probabilities
			W = sum(weights[i] for i in range(N))
			for i in range(N):
				x[i] = weights[i] / W

		# exploitation
		else:

			# choose arm to pull according to distribution x
			chosen_arm = rng.choice(N, p=x)

			# observe loss and update cumulative payout
			loss = 1-payouts[chosen_arm][t]
			simple_payout += payouts[chosen_arm][t]


	# display final vector of probabilities and cumulative reward
	# print('Simple MAB vector of arm probabilities: ', x, '\n', 'Simple MAB payout: ', simple_payout)

	# store cumulative reward for later analysis
	simple_payouts.append(simple_payout)
	

#########
## Average the payout for each arm and return the label of the parking lot

simple_total_payouts = [0 for _ in range(N)]
simple_number_pulls = [0 for _ in range(N)]

# Update total payouts and number of pulls in the algorithm
if explore_or_exploit == 1:
    simple_total_payouts[chosen_arm] += payouts[chosen_arm][t]
    simple_number_pulls[chosen_arm] += 1
else:
    simple_total_payouts[chosen_arm] += payouts[chosen_arm][t]
    simple_number_pulls[chosen_arm] += 1
	    
simple_average_payouts = [p/n if n > 0 else 0 for p, n in zip(simple_total_payouts, simple_number_pulls)]
simple_sorted_arms = sorted(range(N), key=lambda i: -simple_average_payouts[i])
simple_sorted_labels = [labels[i] for i in simple_sorted_arms]

# calculate mean and variance of K cumulative payouts
print('\n', 'Simple MAB average payout: ', mean(simple_payouts), '\n', 'Simple MAB payout variance: ', variance(simple_payouts))	
print(simple_sorted_arms)
print(simple_sorted_labels)
###############################################################

# EXP3 Algorithm

# run algorithm K times
for k in range(K):

	# keep track of cumulative payout for EXP3
	exp3_payout = 0

	# initial vector of arm selection probabilities
	x = [(1/N) for i in range(N)]

	# pick an appropriate value for epsilon
	epsilon = np.sqrt(np.log(N) / (N*T))

	# EXP3 algorithm
	for t in range(T):

		# placeholder vector y used to update x
		y = x.copy()

		# choose arm to pull according to distribution x
		chosen_arm = rng.choice(N, p=x)

		# observe loss--will be either 1 (no payout) or 0 (payout)--and update cumulative payout
		loss = 1 - payouts[chosen_arm][t]
		exp3_payout += payouts[chosen_arm][t]

		# calculate y-value for chosen arm
		y[chosen_arm] = x[chosen_arm] * np.exp(-epsilon * loss / x[chosen_arm])
		
		# update vector of selection probabilities
		y_total = sum(y[i] for i in range(N))
		for i in range(N):
			x[i] = y[i] / y_total


	# display final vector of probabilities and cumulative payout
	# print('EXP3 vector of arm probabilities: ', x, '\n', 'EXP3 payout: ', exp3_payout)

	# store cumulative payout for later analysis
	exp3_payouts.append(exp3_payout)


#########
## Average the payout for each arm and return the label of the parking lot

# Initialize lists
exp3_total_payouts = [0 for _ in range(N)]
exp3_number_pulls = [0 for _ in range(N)]

# Update total payouts and number of pulls in the algorithm
exp3_total_payouts[chosen_arm] += payouts[chosen_arm][t]
exp3_number_pulls[chosen_arm] += 1

# Calculate average payouts and sort arms
exp3_average_payouts = [p/n if n > 0 else 0 for p, n in zip(exp3_total_payouts, exp3_number_pulls)]
exp3_sorted_arms = sorted(range(N), key=lambda i: -exp3_average_payouts[i])
exp3_sorted_labels = [labels[i] for i in exp3_sorted_arms]

# calculate mean and variance of K cumulative payouts
print('\n', 'EXP3 average payout: ', mean(exp3_payouts), '\n', 'EXP3 payout variance: ', variance(exp3_payouts))
print(exp3_sorted_arms)
print(exp3_sorted_labels)
###############################################################

# UCB Algorithm

# lists for storing numbers of pulls, empirical estimates of expected loss, lower confidence bounds for each arm
number_pulls = [0 for i in range(N)]
muhat = [0 for i in range(N)]
lcb = [0 for i in range(N)]

# variable for tracking total payout from arms chosen by UCB
ucb_payout = 0

# start by pulling each arm once and observing losses
for t in range(N):

	chosen_arm = t

	# update number of times arm has been pulled
	number_pulls[chosen_arm] += 1

	# observe arm payout and translate to loss
	loss = 1-payouts[chosen_arm][t]
	ucb_payout += payouts[chosen_arm][t]

	# initial empirical estimate of expected loss is just loss incurred on first pull
	muhat[chosen_arm] = loss

# now start pulling the arm with the smallest lower confidence bound at each round
for t in range(N,T):

	# calculate LCB for each arm
	for i in range(N):

		lcb[i] = muhat[i] - np.sqrt((3 * np.log(t)) / number_pulls[i])# ***formula for lower confidence bound here***

	# choose arm with smallest LCB
	chosen_arm = np.argmin(lcb) # ***index of arm with smallest LCB***

	# observe arm payout and translate to loss, update cumulative payout
	loss = 1 - payouts[chosen_arm][t]
	ucb_payout += payouts[chosen_arm][t]

	# update number of pulls, empirical estimate of expected loss for chosen arm
	number_pulls[chosen_arm] += 1
	muhat[chosen_arm] = ((muhat[chosen_arm] * (number_pulls[chosen_arm] - 1)) + loss) / number_pulls[chosen_arm] # ***update for empirical estimate***


#########
## Average the payout for each arm and return the label of the parking lot

ucb_sorted_arms = sorted(range(N), key=lambda i: -muhat[i])
ucb_sorted_labels = [labels[i] for i in ucb_sorted_arms]

# print empirical estimates for each arm's average payout, cumulative payout from running UCB over K rounds
print('\n', 'UCB empirical estimates: ', muhat, '\n', 'UCB payout: ', ucb_payout)
print(ucb_sorted_arms)
print(ucb_sorted_labels)

###############################################################

# Thompson Sampling Algorithm

# list for storing sampled probabilities of success for each arm
phat = [0 for i in range(N)]

# run algorithm K times
for k in range(K):
	
	# variable for tracking total payout from arms chosen by UCB
	ts_payout = 0

	# lists for storing alpha and beta parameters for each arm's beta distribution
	alpha = [1 for i in range(N)]
	beta = [1 for i in range(N)]

	# algorithm execution
	for t in range(T):

		# sample phat values from beta distributions
		for i in range(N):
			phat[i] = np.random.beta(alpha[i], beta[i]) # which should be -> (math.factorial(alpha[i] - 1) * math.factorial(beta[i] - 1)) / (math.factorial(alpha[i] + beta[i] - 1)) 

		# choose arm with largest phat value
		chosen_arm = np.argmax(phat) # ***index of arm with largest phat***

		# observe arm payout and update cumulative payout
		reward = payouts[chosen_arm][t]
		ts_payout += reward

		# update beta distribution parameters for chosen arm
		if reward == 1:
			alpha[chosen_arm] += 1 # ***parameter update***
			
		else: 
			beta[chosen_arm] += 1 # ***other parameter update***
			

	ts_payouts.append(ts_payout)
	#print(alpha, beta, ts_payout)


#########
## Average the payout for each arm and return the label of the parking lot

# Initialize lists
ts_total_payouts = [0 for _ in range(N)]
ts_number_pulls = [0 for _ in range(N)]

# Update total payouts and number of pulls in the algorithm
ts_total_payouts[chosen_arm] += payouts[chosen_arm][t]
ts_number_pulls[chosen_arm] += 1

# Calculate average payouts and sort arms
ts_average_payouts = [p/n if n > 0 else 0 for p, n in zip(ts_total_payouts, ts_number_pulls)]
ts_sorted_arms = sorted(range(N), key=lambda i: -ts_average_payouts[i])
ts_sorted_labels = [labels[i] for i in ts_sorted_arms]

# calculate mean and variance of K cumulative payouts
print('\n', 'TS average payout: ', mean(ts_payouts), '\n', 'TS payout variance: ', variance(ts_payouts))
print(ts_sorted_arms)
print(ts_sorted_labels)