import numpy as np
# import flopy
# import kle
# from fixed_parameters import *
# import flopy_api
# import flopy_changes
# from flopy_changes import num_outputs, f
from run_final_flopy import num_dims, num_outputs, f, times, p_names
import time


# http://doc.openturns.org/openturns-latest/html/ExamplesGuide/cid2.xhtml#uid99
# http://doc.openturns.org/openturns-0.13.2/doc/html/ReferenceGuide/output/OpenTURNS_ReferenceGuidesu73.html

# initialize
# mf = flopy_api.build_default_model()

# def f( x ):
# 	print "x", x


# 	flopy_changes.change_wel(mf, x)
# 	# flopy_changes.change_upw( mf, kle.compute_kle(x) )

# 	flopy_output = flopy_api.run_model(mf)
# 	heads =  flopy_output.get_data(totim=201.0)
# 	y = np.array([ heads[0, p['row'], p['col']] for p in observed_points ])
# 	print "y", y
# 	return y[:1]


from openturns import *

# Number of input random variables   
# num_dims=10

# NOTE if we have 5 and include whole x we get "null output sample variance" error

class modelePYTHON(OpenTURNSPythonFunction) : 
	# that following method defines the input size (20) and the output size (1)
	def __init__(self) : 
		OpenTURNSPythonFunction.__init__(self,num_dims,num_outputs) 

	# that following method gives the implementation of modelePYTHON 
	def _exec(self,x) :
		return f(x)
		try:
			ret = f(x)
			return ret

		except:
			print "busted", x
			return np.zeros(num_outputs)
			# return [None]


# Create a distribution of dimension n for the input random vector
dists = [Uniform(0,1) for i in range(num_dims)]
Xdist = ComposedDistribution(DistributionCollection( dists ))
vectX = RandomVector(Xdist)
extended_dists = [Uniform(0,1.5) for i in range(num_dims)]
extended_Xdist = ComposedDistribution(DistributionCollection( extended_dists ))
extended_vectX = RandomVector(extended_Xdist)

from numpy import empty, argmin, array, arange, floor, sqrt, linspace
from pylab import ion, figure, semilogy, xlabel, ylabel, bar, legend, xticks, yticks, hist,  scatter, title, plt
#
# Model function
#
myModel = NumericalMathFunction(modelePYTHON())
#
# Output random vector
#
# vectY = RandomVector(myModel, vectX)
#
# Basis of the polynomial chaos (PC) expansion (Hermite polynomials are selected)
#
polyColl = PolynomialFamilyCollection(num_dims)
for i in range(num_dims):
  polyColl[i] = HermiteFactory()
#
enumerateFunction = LinearEnumerateFunction(num_dims)
multivariateBasis = OrthogonalProductPolynomialFactory(polyColl,enumerateFunction)
#
# Definition of the approx. algo.: Least Angle Regression (LAR)
#
basisSequenceFactory = LAR()
fittingAlgorithm = CorrectedLeaveOneOut()
approximationAlgorithm = LeastSquaresMetaModelSelectionFactory(basisSequenceFactory,fittingAlgorithm)
#
# Number of simulations (i.e. simulation budget)
#
N = 100
#
# Initialization of the seed of the random generator
RandomGenerator.SetSeed(77)
#
evalStrategy = LeastSquaresStrategy(LHSExperiment(N),approximationAlgorithm)


# #
# # Parametric study varying the PC degree
# #
# pmax = 4
# Results_tuple = ()
# Relative_errors = empty(pmax)
# p_list = range(1,pmax+1)
# #
# for p in p_list:
# 	#
# 	P = enumerateFunction.getStrataCumulatedCardinal(p)
# 	truncatureBasisStrategy = FixedStrategy(multivariateBasis,P)
# 	#
# 	polynomialChaosAlgorithm = FunctionalChaosAlgorithm (myModel, \
# 	Distribution(Xdist) , AdaptiveStrategy(truncatureBasisStrategy), \
# 	ProjectionStrategy(evalStrategy))
# 	polynomialChaosAlgorithm.run()
# 	#
# 	new_result = polynomialChaosAlgorithm.getResult()

# 	Results_tuple = Results_tuple + (new_result,)
# 	#
# 	Relative_errors[p-1] = array(new_result.getRelativeErrors())[0]
# #
# # Plot the obtained relative errors
# # ion()
# lw = 2.5
# figure(1)
# semilogy(p_list,Relative_errors,linewidth=lw)
# xticks(list(arange(pmax)+1))
# xlabel('PC degree') ; ylabel('Relative corrected leave-one-out error')

# #
# # Identify the most accurate PC expansion
# p_optim = argmin(Relative_errors) + 1
# The_Result = Results_tuple[p_optim-1]
# # #
# print "" ; print "Optimal degree: ", p_optim
# print "" ; print "Relative corrected LOO error:", Relative_errors[p_optim-1]
# print "" ; print "Number of nonzero coefficients:", len(The_Result.getIndices())


# #
# just use one order
# #
p = 3

P = enumerateFunction.getStrataCumulatedCardinal(p)
truncatureBasisStrategy = FixedStrategy(multivariateBasis,P)
#
polynomialChaosAlgorithm = FunctionalChaosAlgorithm (myModel, \
Distribution(Xdist) , AdaptiveStrategy(truncatureBasisStrategy), \
ProjectionStrategy(evalStrategy))
t0 = time.time()
polynomialChaosAlgorithm.run()
t1 = time.time()
building_time = t1-t0
average_run_time_just_mf = np.average(times)
#
The_Result = polynomialChaosAlgorithm.getResult()


#
# Post-processing of the optimal PC expansion
#
ChaosRV = FunctionalChaosRandomVector(The_Result)
#
Mean = ChaosRV.getMean()[0]
StD = sqrt(ChaosRV.getCovariance()[0,0])
print "" ; print "Response mean: ", Mean ; print ""
print "" ; print "Response standard deviation: ", StD ; print ""
#

SU_by_output = []
SUT_by_output = []

for output in range(num_outputs):
	SU = empty(num_dims) ; SUT = empty(num_dims)
	for i in range(num_dims):
	  SU[i] = ChaosRV.getSobolIndex(i,output) # ChaosRV.getSobolIndex(i,j) j in num_output
	  SUT[i] = ChaosRV.getSobolTotalIndex(i, output)

	SUT_by_output.append(SUT)
	SU_by_output.append(SU)
	#
	# Plot the sensitivity indices
	w = 0.4
	figure(2+output)
	b1 = bar((arange(num_dims)+1)-w,SU,width=w,color='#000999')
	b2 = bar((arange(num_dims)+1),SUT,width=w,color='#66FFFF')
	legend((b1[0],b2[0]),('Sobol indices','Total Sobol indices'),'upper left')
	title('output ' + str(output))
	xticks(list(arange(num_dims)+1),(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$',r'$x_8$',r'$x_9$',r'$x_10$'),size=18)
	yticks(list( linspace(0.,1.,5) ))
#
#Plot histogram
truss_model_PC = The_Result.getMetaModel()
#

samplesize = 50
sample_X = vectX.getSample(samplesize)

t0 = time.time()
sample_Y = myModel(sample_X)
t1 = time.time()

sample_YPC = truss_model_PC(sample_X)
t2 = time.time()
asampleYPC = array(sample_YPC).flatten()
#
figure(num_outputs+2)
hist(sample_YPC,normed=True,bins=floor(sqrt(samplesize)))
print "HOW NOW BROWN COW", np.shape(sample_YPC), np.shape(asampleYPC)
#
print "" ; print "Skewness:",  sample_YPC.computeSkewnessPerComponent()[0] ; print ""
print "" ; print "Kurtosis:",  sample_YPC.computeKurtosisPerComponent()[0] ; print ""

average_run_time = (t1-t0)/samplesize
average_surrogate_run_time = (t2-t1)/samplesize
print 'times', t1-t0, t2-t1, (t1-t0)/(t2-t1)
print "building time", building_time


extended_samplesize = 50
extended_sample_X = extended_vectX.getSample(extended_samplesize)

t0 = time.time()
extended_sample_Y = myModel(extended_sample_X)
t1 = time.time()

extended_sample_YPC = truss_model_PC(sample_X)


np.savez("openturns_results.npz", average_run_time_just_mf = average_run_time_just_mf,
						 average_run_time=average_run_time, 
						average_surrogate_run_time=average_surrogate_run_time,
						 building_time=building_time, 
						sample_YPC=sample_YPC, sample_X=sample_X, sample_Y=sample_Y,
						SUT_by_output=SUT_by_output,SU_by_output=SU_by_output)

import json
data_dir = "/home/mikey/Dropbox/pce/agu/data/"
with open(data_dir+'openturns_pce.json','w') as f:
	f.write(json.dumps({
		"p_names": p_names,
		"sample_X": np.array(sample_X).tolist(),
		"sample_Y": np.array(sample_Y).tolist(),
		"sample_YPC": np.array(sample_YPC).tolist(),
		"extended_sample_Y": np.array(extended_sample_Y).tolist(),
		"extended_sample_YPC": np.array(extended_sample_YPC).tolist(),
		"building_time": building_time,
		"average_run_time": average_run_time,
		"average_surrogate_run_time": average_surrogate_run_time,
		"SUT_by_output": np.array(SUT_by_output).tolist(),
		"SU_by_output": np.array(SU_by_output).tolist(),
		}))

for output in range(num_outputs):
	figure(num_outputs+4+output)
	# plt.subplot(1,num_outputs,i+1)
	# figure.title('openturns pce vs complex')
	scatter(sample_Y[:,output], sample_YPC[:,output])

plt.show()
import matplotlib.pyplot as plt
plt.show()

#
# figure(4)
# for i in range(num_outputs):
#     plt.subplot(1,num_outputs,i+1)
#     plt.scatter(new_test_vals[:,i], surrogate_vals[:,i])
#


# import matplotlib.pyplot as plt
# pylab.show()