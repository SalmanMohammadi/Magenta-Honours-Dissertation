import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats

# results - evidence
# x = truth
# composers = truth

# x = pd.read_csv("data/evaluations/turing/orderings.csv", header=None).values[:,[3, 8, 5, 0, 11, 4, 7, 1]]
# results = pd.read_csv("data/evaluations/turing/results_turing.csv", header=None).values
# results_letter = np.empty((8, 8), dtype='str')
# for i in range(len(x)):
# 	for j in range(len(x[i])):
# 		results_letter[i][j] = x[i][j][results[i][j]-1]

# print(results_letter)

# x = pd.read_csv("data/evaluations/human/orderings.csv", header=None).values[:6,[3, 8, 5, 0, 11, 4, 7, 1]]
# composers = pd.read_csv("data/evaluations/human/orderings.csv", header=None).values[6:,[3, 8, 5, 0, 11, 4, 7, 1]]
# results = pd.read_csv("data/evaluations/human/results_human.csv", header=None).values
# print(results)
# print(x)
# print(composers)
# for i in range(len(x)):
# 	for j in range(len(x[i])):
# 		composers[i][j] = composers[i][j][results[i][j]-1]
# 		print(i, j , x[i][j], results[i][j]-1)
# 		results_letter[i][j] = x[i][j][results[i][j]-1]

# anchors = np.empty((6, 8), dtype=int)
# for i in range(len(x)):
# 	for j in range(len(x[i])):

# 		# print(j)
# 		# print(results[i][j])
# 		# results_letter[i][j] = x[i][j][results[i][j]-1]
# 		# print(results[i][j]-1)
# 		# print(x[i][j].lower())
# 		cur = x[i][j].lower()
# 		# print(results[i][j]-1)
# 		# print(cur)
# 		if cur[results[i][j]-1] == 'y':
# 			if cur.count('x') > cur.count(cur[results[i][j]-1]):
# 				anchors[i][j] = x[i][j].find('X')
# 				results_letter[i][j] = 'T'
# 			else:
# 				anchors[i][j] = x[i][j].find('Y')
# 				results_letter[i][j] = 'F' 
# 			# print(results_letter[i][j])
# 		else:
# 			# results_letter[i][j] = cur.count('y') > cur.count(cur[results[i][j]-1])
# 			if cur.count('y') > cur.count(cur[results[i][j]-1]):
# 				anchors[i][j] = x[i][j].find('Y')
# 				results_letter[i][j] = 'T'
# 			else:
# 				anchors[i][j] = x[i][j].find('X')
# 				results_letter[i][j] = 'F' 
# 			print(results_letter[i][j])
# 			if cur[results[i][j]-1] == 'y':
# 				print("x")
# 	        	# print("The anchor is " + str(order.find('X')+1))
# 			else:
# 				print("y")
	        	# print("The anchor is " + str(order.find('Y')+1))

# mistakes = sum(list(z).count('F') for z in results_letter)

# correct = sum(list(z).count('T') for z in results_letter)
# plt.bar(['c', 'f'], [correct, mistakes])
# plt.show()

x = pd.read_csv("data/evaluations/ai/orderings.csv", header=None).values[:8,[3, 8, 5, 0, 11, 4, 7, 1]]
composers = pd.read_csv("data/evaluations/ai/orderings.csv", header=None).values[8:, [3, 8, 5, 0, 11, 4, 7, 1]]
results = pd.read_csv("data/evaluations/ai/results_ai.csv", header=None).values
results_letter = np.empty((8, 8), dtype='str')
for i in range(len(x)):
	for j in range(len(x[i])):
		results_letter[i][j] = x[i][j][results[i][j]-1]

# print("anchor", anchors)
print("results", results_letter)
# print("composer", composers)
# plt.figure(figsize=(10,10))
class_mistakes = {'B':0, 'D':0, 'J':0, 'C':0}
class_corrects = {'B':0, 'D':0, 'J':0, 'C':0}
for i in range(len(results_letter)):
	for j in range(len(results_letter[i])):
		# print(anchors[i][j])
		print(i, j , results_letter[i][j], composers[i][j])
		if results_letter[i][j] == 'X':
			class_corrects[composers[i][j]] += 1
		else:
			class_mistakes[composers[i][j]] += 1

print(class_corrects)
print(class_mistakes)
# labels = class_mistakes.keys()

# _X = np.arange(len(['Beethoven', 'Debussy', 'Bach', 'Chopin']))
# # plt.subplot(211)
# plt.bar(_X - 0.2, class_corrects.values(), 0.4, label='CondRNN')
# plt.bar(_X + 0.2, class_mistakes.values(), 0.4, label='Baseline')
# plt.xticks(_X, labels) # set labels manually
# plt.xlabel("Composer")
# plt.gca().yaxis.grid(True)
# plt.title("Frequency of each model being discriminated against compositional style.")
# plt.ylabel("Number of times discriminated")

# exit()
# results_letter = results_letter.T
# print("results_", results_letter.T)
mistakes = [list(z).count('Y') for z in results_letter]
correct = [list(z).count('X') for z in results_letter]
_X = np.arange(1, len(mistakes)+1)
# mistakes = sum(list(z).count('Y') for z in results_letter)
# correct = sum(list(z).count('X') for z in results_letter)
# expected_values = mistakes+cnp.orrect/2
# obv =  [correct, mistakes]
print(mistakes, np.mean(mistakes), np.std(mistakes))
print(correct, np.mean(correct), np.std(correct))
# print(obv)
# exp = [expected_values, expected_values]
# plt.subplot(212)
plt.bar(_X - 0.2, mistakes, 0.4, label='Baseline')
plt.bar(_X + 0.2, correct, 0.4, label='CondRNN')
plt.xlabel("Experiment No.")
plt.ylabel("Number of times discriminated")
plt.title("Frequency of answers for discriminating model samples for compositional style.")

plt.gca().yaxis.grid(True)
plt.legend()
plt.show()
exit()
q1_chi_square = stats.chisquare(obv, exp)
print(q1_chi_square)
x = np.linspace(0, 20, 10000)
# fig = plt.figure(figsize=(10, 5))

degrees_freedom = 1
plt.subplot(313)
plt.plot(x, stats.chi2.pdf(x, degrees_freedom), label=r'$df=%i$' % degrees_freedom)

plt.xlim(-.1, 20)
plt.legend()
plt.ylim(0, 0.5)

plt.axvline(x=3.85, color='blue', label='critical value, p=0.05')
# plt.axvline(x=6.63, color='blue', label='critical value, p=0.01')
plt.axvline(x=q1_chi_square.statistic, color='red', label='chi square')
plt.xlabel('$\chi^2$')
plt.ylabel(r'$f(\chi^2)$')
plt.title(r'$\chi^2\ \mathrm{Distribution}$')

plt.legend()
plt.show()

# print(sum(list(z).count('X') for z in results_letter))
# print(sum(list(z).count('Y') for z in results_letter))