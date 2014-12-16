# import numpy as np
# import flopy
# import kle
# from fixed_parameters import *

# from run_final_flopy import num_dims, num_outputs, f, p_names

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

from multi_resolution_run_final_flopy import num_dims, num_outputs, mf_model

# ,nlay,nrow,ncol
for dims in [(1,5,10),(1,10,20),(1,20,40),(1,40,80)]:
    nlay,nrow,ncol = dims
    this_model = mf_model(nlay,nrow,ncol)
    def f(x):
        return this_model.f(x)


    model = CppModel( f )


    # =====================================
    # variation 2 from pce_regression_example
    # =====================================


    from models.odes.odes import RandomOscillator
    from polynomial_chaos_cpp import PolynomialChaosExpansion, LegendrePolynomial1D,\
        initialise_homogeneous_pce
    from model_cpp import RandomVariableTransformation
    from utilities.generalised_sparse_grid_tools import get_quadrature_rule_1d,\
        convert_homogeneous_sparse_grid_to_pce, \
        HierarchicalSurplusDimensionRefinementManager
    from pce_regression_example import \
        define_random_variable_transformation_oscillator, plot_sensitivities

    from utilities.generalised_sparse_grid_tools import build_generalised_sparse_grid

    max_num_points = 100
    rv_trans = define_homogeneous_input_space( 'uniform', num_dims, 
                                       ranges = [0,1.] )

    quad_type = 'clenshaw-curtis'
    quadrature_rule_1d, orthog_poly_1d = get_quadrature_rule_1d( quad_type )
    max_level = numpy.iinfo( numpy.int32 ).max
    max_level_1d = numpy.iinfo( numpy.int32 ).max


    num_pts = 20
    rng = numpy.random.RandomState( 0 )
    test_pts = rng.uniform( 0, 1., ( rv_trans.num_dims(), num_pts ) )
    # if using map_from_canonical_distributions, use canonical eg -1,1 uniform
    # test_pts = rv_trans.map_from_canonical_distributions( test_pts )




    t0 = time.time()
    sg = build_generalised_sparse_grid( quadrature_rule_1d, orthog_poly_1d,
                                        rv_trans, model, max_num_points,
                                        tpqr = None, rm = None, tolerance = 1e-12, 
                                        max_level = max_level, 
                                        max_level_1d = max_level_1d,
                                        verbosity = 0,
                                        test_pts = None, 
                                        test_vals = None, 
                                        breaks = None )[0]


    # TODO ask John whether necessary or not, it's in build_generalised_sparse_grid
    pce = PolynomialChaosExpansion()
    pce.set_random_variable_transformation( rv_trans )
    sg.convert_to_polynomial_chaos_expansion( pce, 0 )


    from serialization_cpp import save_pce
    save_pce(pce, 'multi_res_pce_%i_%i.dat' %(nrow,ncol), 1) # 1 for text, 0 for binary (John says binary is risky)



    # t0 = time.time()
    # test_vals = model.evaluate_set( test_pts )  
    # t1 = time.time()
    # complex_runtime = t1-t0


    # t1 = time.time()
    # surrogate_vals = pce.evaluate_set( test_pts )
    # t2 = time.time()

    # print "building time", t1-t0
    # print "surrogate time", t2-t1
    # print "complex time", complex_runtime

    # # print "norm of residual:", numpy.linalg.norm(new_test_vals - surrogate_vals), "meters"
    # num_outputs = numpy.shape(test_vals)[1]
    # for i in range(num_outputs):
    #     plt.subplot(1,num_outputs,i+1)
    #     plt.scatter(test_vals[:,i], surrogate_vals[:,i])
    # plt.show()

    # print 'error', numpy.linalg.norm( test_vals.squeeze() -surrogate_vals.squeeze() ) / numpy.sqrt( test_vals.shape[0] )

    # plot_sensitivities( pce )
