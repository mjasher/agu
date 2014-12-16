import numpy
try:
    from scipy.misc import comb as nchoosek
except:
    print 'could not import scipy'
import time
import pickle
import os

from models.algebraic.genz_model import GenzModel
from polynomial_chaos_cpp import PolynomialChaosExpansion, LegendrePolynomial1D,\
    OrthogPolyVector, TensorProductBasis, HermitePolynomial1D,\
    MultipleSolutionLinearModelCrossValidationIterator, \
    PCECrossValidationIterator, initialise_homogeneous_pce
from math_tools_cpp import hypercube_map, set_hypercube_domain, lhs
from compressed_sensing_cpp import  OMPSolver, LARSSolver
from sparse_grid_cpp import GeneralisedSparseGrid
from indexing_cpp import SubSpaceVector, PolyIndexVector, PolynomialIndex, \
    get_anisotropic_hyperbolic_indices
from math_tools_cpp import compute_hyperbolic_indices
from refinement_manager_cpp import PCERefinementManager, \
    HierarchicalSurplusDimensionRefinementManager
from model_cpp import RandomVariableTransformation
from quadrature_rules_cpp import ClenshawCurtisQuadRule1D, \
    GaussPattersonQuadRule1D, GaussHermiteQuadRule1D, GenzKeisterQuadRule1D,\
    GaussJacobiQuadRule1D, UniformLejaQuadRule1D, NormalLejaQuadRule1D, \
    GaussLegendreQuadRule1D, LagrangePolynomialBasis, QuadRule1DVector, \
    TensorProductQuadratureRule


from serialization_cpp import save_pce

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


def build_generalised_sparse_grid( quadrature_rule_1d, orthog_poly_1d, 
                                   rv_trans, f, max_num_points,
                                   tpqr = None, rm = None, tolerance = 0., 
                                   max_level = None, max_level_1d = None,
                                   verbosity = 0,
                                   test_pts = None, test_vals = None, 
                                   test_mean = None, test_variance = None, 
                                   breaks = None ):

    start_time = time.time()

    num_dims = rv_trans.num_dims()
    basis = LagrangePolynomialBasis()
    if tpqr is None:
        tpqr = get_tensor_product_quadrature_rule_single( num_dims, 
                                                          quadrature_rule_1d,
                                                          basis )

    sg = GeneralisedSparseGrid()
    sg.verbosity( verbosity )
    sg.set_interpolation_type( 0 )

    convert = False
    if rm is None:
        pce = PolynomialChaosExpansion()
        pce.set_random_variable_transformation( rv_trans )
        sg.set_polynomial_chaos_expansion( pce )
        rm = PCERefinementManager()
    elif test_mean is not None or test_variance is not None:
        convert = True

    rm.set_grid( sg )
    rm.set_min_level( 0 )
    sg.set_interpolation_type( 0 )
    rm.set_tolerance( tolerance )
    rm.set_max_num_function_evaluations( max_num_points )
    if max_level is None:
        max_level = 500
    rm.set_max_level( max_level )

    if max_level_1d is not None:
        rm.set_max_level_1d( max_level_1d )

    break_cnt = 0
    if test_pts is not None and test_vals is None:
        test_vals = f.evaluate_set( test_pts )[:,0]

    if breaks is None:
        breaks = numpy.array( [max_num_points] )

    l2_errors = []
    num_pts = []
    mean_errors = []
    var_errors = []

    sg.initialise( f, rv_trans, tpqr, rm, 0 )
    sg.initialise_build();

    while ( True ):
        sg.prioritise();

        # mja assuming pce exists already, save at break points
        
        num_eval = sg.num_function_evaluations()
        if break_cnt < breaks.shape[0] and num_eval >= breaks[break_cnt]:
            now_time = time.time()-start_time
            print "time", num_eval, now_time, now_time/num_eval
            save_pce(pce, 'pce_%i.dat' %num_eval, 1) # 1 for text, 0 for binary (John says binary is risky)
            break_cnt += 1



        if ( test_pts is not None and break_cnt < breaks.shape[0] 
             and sg.num_function_evaluations() >= breaks[break_cnt] ):
            print '--------'
            # use all function evaluations to compute error
            sg_full = GeneralisedSparseGrid()
            sg_full.copy( sg )
            sg_full.complete_build()
            
            pred_vals = sg_full.evaluate_set( test_pts, [0] )
            l2_err = l2_error( test_vals, pred_vals )
            print 'num func evals: ', sg_full.num_function_evaluations()
            print 'l2 error: ', l2_err
            l2_errors.append( l2_err )
            num_pts.append( sg_full.num_function_evaluations() )
            if convert:
                pce = convert_homogeneous_sparse_grid_to_pce( sg_full,
                                                              orthog_poly_1d )
            if test_mean is not None:
                mean_errors.append( abs( test_mean - pce.mean()[0] ) )
                print 'mean error: ', mean_errors[-1]
                print pce.mean()[0], test_mean
            if test_variance is not None:
                var_errors.append( abs ( test_variance - pce.variance()[0] ) )
                print 'variance error: ', var_errors[-1]
                print pce.variance()[0], test_variance
            print 'mean=%1.16e, variance=%1.16e' %( pce.mean()[0], pce.variance()[0] )
            print '*********'
            break_cnt += 1
            # for some reason copy sets rm._max_num_function_evaluations to inf
            rm.set_max_num_function_evaluations( max_num_points )

        if ( not sg.grid_warrants_refinement() ): break;
        sg.refine();
        
    sg.complete_build()

    if ( test_pts is not None ):
        # use all function evaluations to compute error
        sg_full = GeneralisedSparseGrid()
        sg_full.copy( sg )
        sg_full.complete_build()
        pred_vals = sg_full.evaluate_set( test_pts, [0] )
        l2_err = l2_error( test_vals, pred_vals )
        print 'num func evals: ', sg_full.num_function_evaluations()
        print 'l2 error: ', l2_err
        l2_errors.append( l2_err )
        num_pts.append( sg_full.num_function_evaluations() )
        if convert:
            pce = convert_homogeneous_sparse_grid_to_pce( sg_full,
                                                          orthog_poly_1d )
        if test_mean is not None:
            mean_errors.append( abs( test_mean - pce.mean()[0] ) )
        if test_variance is not None:
            var_errors.append( abs ( test_variance - pce.variance()[0] ) )
    
    result = ( sg, )# , needed so I can append
    if test_pts is not None:
        result = result + ( numpy.asarray( l2_errors ),numpy.asarray( num_pts ),)
    if test_mean is not None:
        result = result + ( numpy.asarray( mean_errors ), )
    if test_variance is not None:
        result = result + ( numpy.asarray( var_errors ), )
    return result