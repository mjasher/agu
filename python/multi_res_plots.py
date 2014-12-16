import pylab, numpy
import os, re, json

from models.base_models import CppModel
from pce_regression_example import get_sensitivities, get_interactions, plot_sensitivities
from utilities.generalised_sparse_grid_tools import load_pce


from run_final_flopy import num_outputs, p_names, num_dims, f


# validate
###############################
num_pts = 50
rng = numpy.random.RandomState( 0 )
test_pts = rng.uniform( -1, 1., ( num_dims, num_pts ) )

model = CppModel( f )

test_vals = model.evaluate_set( test_pts ) 

sensitivity_data = {}
error_data = {"test_vals": test_vals.tolist()}

break_re = re.compile('pce_(\d+)\.dat')
# data_dir = "/home/mikey/Dropbox/pce/agu/python/multi_resolution_results/"
data_dir = "/home/mikey/Dropbox/pce/agu/python/40_80/"
files = [ break_re.match(f).group(0) for f in os.listdir(data_dir) if break_re.match(f)]
for f in files:
	res = break_re.match(f).group(1)

	loaded = load_pce(data_dir+f, 1)
	# print loaded.get_coefficients().shape
	# print loaded.evaluate_set(test_pts)
	# plot_sensitivities(loaded)

	surrogate_vals = loaded.evaluate_set( test_pts )

	error = numpy.linalg.norm( test_vals.squeeze() -surrogate_vals.squeeze() ) / numpy.sqrt( test_vals.shape[0] )
	print 'error', f, error

	error_data[res] = {}

	# make json (for d3.js visualization)
	###############################

	sensitivity_data[res] = {}

	for qoi in range(num_outputs):

		error = numpy.linalg.norm( test_vals[:,qoi] -surrogate_vals[:,qoi] ) / numpy.sqrt( test_vals.shape[0] )

		error_data[res][qoi] =  {"surrogate_vals": surrogate_vals[:,qoi].tolist(), "error": error }

		me, te, ie = get_sensitivities( loaded, qoi )
		interaction_values, interaction_terms = get_interactions( loaded, qoi )

		# calculate the dimension-wise joint effects
		# from utilities.visualization.plot_interaction_values
		# interaction_values /= interaction_values.sum()
		# truncation_pct = 0.95
		# max_slices = 5
		# rv = 'z'
		# labels = []
		# partial_sum = 0.
		# for i in xrange( interaction_values.shape[0] ):
		# 	if partial_sum < truncation_pct and i < max_slices:
		# 		l = '($'
		# 		for j in xrange( len( interaction_terms[i] )-1 ):
		# 			l += '%s_{%d},' %(rv,interaction_terms[i][j]+1)
		# 		l+='%s_{%d}$)' %(rv,interaction_terms[i][-1]+1)
		# 		# labels.append( l )
		# 		labels.append( p_names[interaction_terms[i][-1]] )
		# 		partial_sum += interaction_values[i]
		# 	else:
		# 		break
		# interaction_values = interaction_values[:i+1]
		# if abs( partial_sum - 1. ) > 10 * numpy.finfo( numpy.double ).eps:
		# 	labels.append( 'other' )
		# 	interaction_values[-1] = 1. - partial_sum


		interaction_values /= interaction_values.sum()

		interaction_labels = [ '-'.join([p_names[t] for t in ts]) for ts in interaction_terms]

		sensitivity_data[res][qoi] = {}

		sensitivity_data[res][qoi]["me"] = me.tolist()
		sensitivity_data[res][qoi]["te"] = te.tolist()
		sensitivity_data[res][qoi]["ie"] = ie.tolist()
		sensitivity_data[res][qoi]["interaction_values"] = { "title": "interaction_values", "values": interaction_values.tolist(), "labels": interaction_labels}
		# sensitivity_data[res][qoi]["interaction_values"] = [v.tolist() for v in interaction_values]
		# sensitivity_data[res][qoi]["interaction_terms"] = [v.tolist() for v in interaction_terms]

with open('/home/mikey/Dropbox/pce/agu/data/multi_res_40_80.json', 'w') as json_f:
	json_f.write(json.dumps({"sensitivity_data": sensitivity_data, "error_data": error_data}))




def pie():
	# make a square figure and axes
	pylab.figure(1, figsize=(6,6))
	ax = pylab.axes([0.1, 0.1, 0.8, 0.8])

	# The slices will be ordered and plotted counter-clockwise.
	labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
	fracs = [15, 30, 45, 10]
	# explode=(0, 0.05, 0, 0)

	pylab.pie(fracs, 
		# explode=explode, 
		labels=labels,
	    autopct='%1.1f%%', shadow=True, startangle=90)
	                # The default startangle is 0, which would start
	                # the Frogs slice on the x-axis.  With startangle=90,
	                # everything is rotated counter-clockwise by 90 degrees,
	                # so the plotting starts on the positive y-axis.

	pylab.title('Raining Hogs and Dogs', bbox={'facecolor':'0.8', 'pad':5})

	pylab.show()

