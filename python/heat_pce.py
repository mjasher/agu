# import numpy as np
# import flopy
# import kle
# from fixed_parameters import *

from run_final_flopy import num_dims, num_outputs, f
import time
import numpy
from models.base_models import CppModel
from model_cpp import define_homogeneous_input_space
from quadrature_rules_cpp import ClenshawCurtisQuadRule1D, LagrangePolynomialBasis, QuadRule1DVector, TensorProductQuadratureRule
from sparse_grid_cpp import AdaptiveSparseGrid, GeneralisedSparseGrid
from refinement_manager_cpp import PCERefinementManager
from polynomial_chaos_cpp import LegendrePolynomial1D, OrthogPolyVector, \
    TensorProductBasis, PolynomialChaosExpansion, HermitePolynomial1D 
import matplotlib.pyplot as plt
from pce_regression_example import plot_sensitivities

def get_tensor_product_quadrature_rule_single( num_dims, 
                                               quadrature_rule_1d,
                                               basis ):
    q = QuadRule1DVector()
    q.push_back( quadrature_rule_1d )
    q[0].set_basis( basis )
    dimension_list = [ numpy.arange( num_dims, dtype=numpy.int32 ) ]
    
    tpqr = TensorProductQuadratureRule()
    tpqr.set_quadrature_rules( q, dimension_list, num_dims )
    
    return tpqr

model = CppModel( f )
rv_trans = define_homogeneous_input_space( 'uniform', num_dims, 
                                       ranges = [0,1.] )

# =====================================
# variation 1 from heat_flopy_steady_state
# =====================================


# rng = numpy.random.RandomState( )
# # define quadrature rule
# quadrature_rule_1d = ClenshawCurtisQuadRule1D()
# basis = LagrangePolynomialBasis()
# tpqr = get_tensor_product_quadrature_rule_single (num_dims, 
#                                                   quadrature_rule_1d,
#                                                   basis )

# # build sparse grid
# sg = GeneralisedSparseGrid()
# rm = PCERefinementManager()
# pce = PolynomialChaosExpansion()
# pce.set_random_variable_transformation( rv_trans )
# sg.set_polynomial_chaos_expansion( pce )
# rm.set_max_num_function_evaluations( numpy.iinfo( numpy.int32 ).max ) 
# # rm.set_tolerance( 10*self.eps )
# eps = 2.2e-14
# rm.set_tolerance( 10*eps )
# rm.set_max_level( 6 )
# rm.set_grid( sg )
# sg.initialise( model, rv_trans, tpqr, rm )
# sg.build()



# new_test_pts = rng.uniform( 0, 1., ( num_dims, 100 ) )
# new_test_vals = model.evaluate_set( new_test_pts )
# surrogate_vals = sg.evaluate_set( new_test_pts )

# print "norm of residual:", numpy.linalg.norm(new_test_vals - surrogate_vals), "meters"

# num_outputs = np.shape(new_test_vals)[1]
# for i in range(num_outputs):
#     plt.subplot(1,num_outputs,i+1)
#     plt.scatter(new_test_vals[:,i], surrogate_vals[:,i])
# plt.show()

# plot_sensitivities( pce )
# plot_sensitivities( sg )


# =====================================
# variation 2 from pce_regression_example
# =====================================


from models.odes.odes import RandomOscillator
from polynomial_chaos_cpp import PolynomialChaosExpansion, LegendrePolynomial1D,\
    initialise_homogeneous_pce
from model_cpp import RandomVariableTransformation
from utilities.generalised_sparse_grid_tools import get_quadrature_rule_1d,\
    convert_homogeneous_sparse_grid_to_pce, build_generalised_sparse_grid,\
    HierarchicalSurplusDimensionRefinementManager
from pce_regression_example import \
    define_random_variable_transformation_oscillator, plot_sensitivities

max_num_points = 100
# rv_trans = define_random_variable_transformation_oscillator()
rv_trans = define_homogeneous_input_space( 'uniform', num_dims, 
                                   ranges = [0,1.] )

# model = RandomOscillator()
quad_type = 'clenshaw-curtis'
quadrature_rule_1d, orthog_poly_1d = get_quadrature_rule_1d( quad_type )
rm = None
max_level = numpy.iinfo( numpy.int32 ).max
max_level_1d = numpy.iinfo( numpy.int32 ).max

t0 = time.time()
sg = build_generalised_sparse_grid( quadrature_rule_1d, orthog_poly_1d,
                                    rv_trans, model, max_num_points,
                                    tpqr = None, rm = rm, tolerance = 1e-12, 
                                    max_level = max_level, 
                                    max_level_1d = max_level_1d,
                                    verbosity = 0,
                                    test_pts = None, 
                                    test_vals = None, 
                                    breaks = None )[0]

pce = PolynomialChaosExpansion()
pce.set_random_variable_transformation( rv_trans )
sg.convert_to_polynomial_chaos_expansion( pce, 0 )

t1 = time.time()
building_time = t1-t0
# sample the surrogate to compute l2 error   

num_pts = 10
rng = numpy.random.RandomState( 0 )
test_pts = rng.uniform( -1., 1., ( rv_trans.num_dims(), num_pts ) )
test_pts = rv_trans.map_from_canonical_distributions( test_pts )

t0 = time.time()
test_vals = model.evaluate_set( test_pts )  
t1 = time.time()
surrogate_vals = pce.evaluate_set( test_pts )
t2 = time.time()

print 'times', t1-t0, t2-t1, (t1-t0)/(t2-t1)
print "building time", building_time

# print "norm of residual:", numpy.linalg.norm(new_test_vals - surrogate_vals), "meters"
num_outputs = numpy.shape(test_vals)[1]
for i in range(num_outputs):
    plt.subplot(1,num_outputs,i+1)
    plt.scatter(test_vals[:,i], surrogate_vals[:,i])
plt.show()

print 'error', numpy.linalg.norm( test_vals.squeeze() -surrogate_vals.squeeze() ) / numpy.sqrt( test_vals.shape[0] )

plot_sensitivities( pce )
