import numpy
from scipy.misc import comb as nchoosek
from polynomial_chaos_cpp import PolynomialChaosExpansion, LegendrePolynomial1D,\
    initialise_homogeneous_pce
from model_cpp import RandomVariableTransformation
from indexing_cpp import PolyIndexVector

from models.algebraic.genz_model import GenzModel
from models.odes.odes import RandomOscillator

def get_sensitivities( pce, qoi = 0 ):
    main_effects = numpy.zeros( ( pce.dimension() ), numpy.double )
    total_effects = numpy.zeros( ( pce.dimension() ), numpy.double )
    interaction_effects = numpy.zeros( ( pce.dimension()+1 ), numpy.double )
    var = pce.variance()[0]
    indices = PolyIndexVector()
    pce.get_basis_indices( indices )
    coeff = pce.get_coefficients()[:,qoi]
    for index in indices:
        # calculate contibution to variance of the index
        var_contribution = \
            coeff[index.get_array_index()]**2*pce.l2_norm( index )
        # get number of dimensions involved in interaction, also known 
        # as order
        order = index.num_effect_dims()
        # update interaction effects
        interaction_effects[order] += var_contribution
        # update main effects
        if ( order == 1 ):
            dim = index.dimension_from_array_index( 0 )
            main_effects[dim] += var_contribution
        # update total effects
        for i in xrange( order ):
            dim = index.dimension_from_array_index( i )
            total_effects[dim] += var_contribution
    main_effects /= var
    total_effects /= var
    interaction_effects /= var
    return main_effects, total_effects, interaction_effects[1:]

def get_interactions( pce, qoi = 0 ):
    interactions = dict()
    indices = PolyIndexVector()
    pce.get_basis_indices( indices )
    coeff = pce.get_coefficients()[:,qoi]
    for index in indices:
        var_contribution = \
            coeff[index.get_array_index()]**2*pce.l2_norm( index )
        key = ()
        for i in xrange( index.num_effect_dims() ):
            dim = index.dimension_from_array_index( i )
            key += (dim,)
        if len( key ) > 0 and interactions.has_key( key ):
            interactions[key] += var_contribution
        elif len( key ) > 0:
            interactions[key] = var_contribution

    # convert interactions to a list for easy sorting
    interaction_terms = []
    interaction_values = []
    for key in interactions.keys():
        interaction_terms.append( numpy.array( key, numpy.int32 ) )
        interaction_values.append( interactions[key] )

    # sort interactions in descending order
    I = numpy.argsort( interaction_values )
    values_tmp = numpy.empty( ( len( I ) ), numpy.double )
    terms_tmp = []
    for i in xrange( len( I ) ):
        values_tmp[i] = interaction_values[I[i]]
        terms_tmp.append( interaction_terms[I[i]] )
    # assert numpy.allclose( numpy.sum( values_tmp ), pce.variance() ) # mja
    values_tmp /= pce.variance()[0]
    return values_tmp[::-1], terms_tmp[::-1]
  
def construct_data():
    num_dims = 6
    model = RandomOscillator()
    rv_trans = define_random_variable_transformation_oscillator()
    num_pts = 1000
    #rng = numpy.random.RandomState( 0 )
    #pts = rng.uniform( -1., 1., ( num_dims, num_pts ) )
    pts = numpy.random.uniform( -1., 1., ( num_dims, num_pts ) )
    #from math_tools_cpp import latin_hypercube_design
    #pts = latin_hypercube_design( num_pts, num_dims, seed ) # returns points on [0,1] but need pts on [-1,1]
    #pts = 2*pts-1.
    pts = rv_trans.map_from_canonical_distributions( pts )
    vals = model.evaluate_set( pts )                  
    numpy.savetxt( 'pts.txt', pts, delimiter = ',' )
    numpy.savetxt( 'vals.txt', vals, delimiter = ',' )
    model.rv_trans = rv_trans
    return model

def find_largest_degree_overdetermined( num_dims , num_pts ):
     # find degree of PCE
    degree = 1
    while ( True ):
        num_basis_terms = nchoosek( degree + num_dims, num_dims )
        if ( num_basis_terms > num_pts ):
            break
        degree += 1
    degree -= 1
    return degree

def build_pce_from_file_data( pts_filename, vals_filename, rv_trans, 
                              degree = None ):
    # Must be a ( num_dims x num_pts ) matrix
    pts = numpy.loadtxt( pts_filename, delimiter = ',' )
    # must be a ( num_pts x 1 ) vector
    vals = numpy.loadtxt( vals_filename, delimiter = ',' )
    vals = vals.reshape( vals.shape[0], 1 )

    ranges = rv_trans.get_distribution_ranges()
    assert not numpy.any( numpy.min( pts, axis = 1 ) < ranges[::2] )
    assert not numpy.any( numpy.max( pts, axis = 1 ) > ranges[1::2] )

    #I = numpy.random.RandomState(seed).permutation( pts.shape[1] )[:num_subsamples]
    #pts = pts[:,I]
    #vals = vals[I,:]

    #import pylab; pylab.plot( pts[0,:], pts[1,:], 'o' ); pylab.show()
    return build_pce( pts, vals, rv_trans, degree )

def build_pce( pts, vals, rv_trans, degree = None ):
    
    num_dims, num_pts = pts.shape 
    if degree is None:
        degree = find_largest_degree_overdetermined( num_dims, num_pts )

    # define the parameters of the PCE
    pce = PolynomialChaosExpansion()
    pce.set_random_variable_transformation( rv_trans )
    pce.define_isotropic_expansion( degree, 1. )

    # form matrices needed for normal equations 
    V, build_vals = pce.build_linear_system( pts, vals, 
                                             False )
    assert V.shape[1] <= V.shape[0]

    # Solve least squares to find PCE coefficients    
    coeff = numpy.linalg.solve( numpy.dot( V.T, V ), 
                                numpy.dot( V.T, build_vals ) )
    pce.set_coefficients( coeff.reshape( coeff.shape[0], 1 ) )

    return pce

def compute_pce_rmse_error_from_data( pce, pts, truth_vals ):
    rmse = numpy.linalg.norm( truth_vals.squeeze() - 
         pce.evaluate_set( pts ).squeeze() ) / numpy.sqrt( truth_vals.shape[0] )
    #rmse /= numpy.std( truth_vals )
    print 'RMSE:', rmse

def compute_pce_rmse_error( pce, model, rv_trans ):
    num_dims = rv_trans.num_dims()
    num_pts = 10000
    rng = numpy.random.RandomState( 0 )
    test_pts = rng.uniform( -1., 1., ( num_dims, num_pts ) )
    test_pts = rv_trans.map_from_canonical_distributions( test_pts )
    test_vals = model.evaluate_set( test_pts )                  
    compute_pce_rmse_error_from_data( pce, test_pts, test_vals )


from utilities.visualisation import *
def plot_sensitivities( pce ):
    me, te, ie = get_sensitivities( pce )
    interaction_values, interaction_terms = get_interactions( pce )
    
    show = False
    fignum = 1
    filename = 'individual-interactions.png'
    plot_interaction_values( interaction_values, interaction_terms, 
                             title = 'Sobol indices', truncation_pct = 0.95, 
                             filename = filename, show = show,
                             fignum = fignum, rv = r'\xi' )
    fignum += 1
    filename = 'dimension-interactions.png'
    plot_interaction_effects( ie, title = 'Dimension-wise joint effects', 
                              truncation_pct = 0.95, filename = filename, 
                              show = show, fignum = fignum )
    fignum += 1
    filename = 'main-effects.png'
    plot_main_effects( me, truncation_pct = 0.95, 
                       title = 'Main effect sensitivity indices', 
                       filename = filename, show = show, fignum = fignum  )
    fignum += 1
    filename = 'total-effects.png'
    plot_total_effects( te, truncation_pct = 0.95, 
                        title = 'Total effect sensitivity indices', 
                        filename = filename, show = show, fignum = fignum,
                        rv = r'\xi'   )
    fignum += 1

    pylab.show()

def define_random_variable_transformation_oscillator():
    rv_trans = RandomVariableTransformation()
    dist_types = ['uniform']*6
    means = numpy.zeros( ( 6 ), numpy.double )    # dummy for uniform
    std_devs = numpy.zeros( ( 6 ), numpy.double ) # dummy for uniform
    ranges = numpy.array( [0.08,0.12,0.03,0.04,0.08,0.12,0.8,1.2,
                           0.45,0.55,-0.05,0.05], numpy.double )
    rv_trans.set_random_variables( dist_types, ranges, means, std_devs )
    return rv_trans

from model_cpp import Model, define_homogeneous_input_space
from math_tools_cpp import cartesian_product_int
class SobolFunction(Model):
    def __init__( self ):
        Model.__init__( self ) # must call c++ object wrapper init function
        self.a = numpy.array( [1.,2.,5.,10.,20.,50.,100.,500.] )
    
    def evaluate( self, x ):
        x = x.squeeze()
        assert x.ndim == 1
        num_dims = x.shape[0]
        result = numpy.prod( (numpy.absolute( 4.*x - 2. ) + self.a[:num_dims])/
                             (1+self.a[:num_dims]))
        if result.shape == ():
            result = numpy.array( [result] )
        return result

    

    def sobol_indices( self, num_dims ):

        partial_var =  1. / ( 3.*(1.+self.a[:num_dims])**2 )
        total_var = numpy.prod( partial_var + 1. ) -1.
        sobol_indices = dict()
        
        sets = [numpy.array( [0, 1 ], numpy.int32 )]*num_dims
        indices  = cartesian_product_int( sets, 1 )

        sobol_indices = []
        for i in xrange( indices.shape[1] ):
            index = indices[:,i]
            I = numpy.where( index==1 )[0]
            if not I.shape[0] == 0:
                sobol_index = numpy.prod( partial_var[I] ) / total_var
                sobol_indices.append( sobol_index )
        assert numpy.allclose( numpy.sum( sobol_indices ), 1., 
                               atol = 1e-15, rtol = 1e-15 )
        
        return indices, sobol_indices

    def variance( self, num_dims ):
        partial_var =  1. / ( 3.*(1.+self.a[:num_dims])**2 )
        total_var = numpy.prod( partial_var + 1. ) -1.
        return total_var
    

class IshigamiFunction(Model):
    def __init__( self ):
        Model.__init__( self ) # must call c++ object wrapper init function
        self.num_dims = 3
        self.A = 7.
        self.B = 0.1

    def evaluate( self, x ):
        x = x.squeeze()
        assert x.ndim == 1
        assert x.shape[0] == self.num_dims
        return numpy.array( [numpy.sin( x[0] ) + 7.*numpy.sin( x[1] )**2 + 
                             0.1*x[2]**4*numpy.sin( x[0] )] )

    def sobol_indices( self ):
        return None, numpy.array( [(1.+self.B*numpy.pi**4/5.)**2/2., 
                                   self.A**2/8., 0., 0., 
                                   self.B**2*numpy.pi**8*8/225., 0., 0.] )

    
    def variance( self ):
        tmp, sobol_indices = self.sobol_indices()
        return numpy.sum( sobol_indices )
    
    
    

def test_sobol_function_sensitivities():
    num_dims = 2
    rv_trans = define_homogeneous_input_space( 'uniform', num_dims, 
                                               ranges =  [0.,1.] )
    model = SobolFunction()
    indices, sobol_indices = model.sobol_indices( num_dims )
    total_var = model.variance( num_dims )
    #print sobol_indices
    #print total_var

    num_pts = 1000
    degree = 20
    rng = numpy.random.RandomState( 0 )
    pts = rng.uniform( -1., 1., ( num_dims, num_pts ) )
    pts = rv_trans.map_from_canonical_distributions( pts )
    vals = model.evaluate_set( pts )
    pce = build_pce( pts, vals, rv_trans, degree )
    filename = 'sobol-pce.dat'
    save_pce( pce, filename, 1 )
    compute_pce_rmse_error( pce, model, rv_trans )
    #plot_sensitivities( pce )
    from utilities.visualisation import plot_surface_from_function
    #plot_surface_from_function( model.evaluate_set )
    #plot_surface_from_function( pce.evaluate_set )
    assert numpy.allclose( total_var, 10./81. )

def test_ishigami_sensitivities():
    num_dims = 3
    rv_trans = define_homogeneous_input_space( 'uniform', num_dims, 
                                               ranges =  [-numpy.pi,numpy.pi] )
    model = IshigamiFunction()
    indices, sobol_indices = model.sobol_indices()
    total_var = model.variance()
    #print sobol_indices
    print 'var', total_var

    num_pts = 10000
    degree = 15
    rng = numpy.random.RandomState( 0 )
    pts = rng.uniform( -1., 1., ( num_dims, num_pts ) )
    pts = rv_trans.map_from_canonical_distributions( pts )
    vals = model.evaluate_set( pts )
    pce = build_pce( pts, vals, rv_trans, degree )
    compute_pce_rmse_error( pce, model, rv_trans )
    sobol_indices, terms = get_interactions( pce )

    me, te, ie = get_sensitivities( pce )
    assert numpy.allclose( te, numpy.array( [sobol_indices[1]+sobol_indices[2],
                                             sobol_indices[0],sobol_indices[2]]))

    interaction_values, interaction_terms = get_interactions( pce )
    
    fignum = 1
    filename = 'individual-interactions.png'
    plot_interaction_values( interaction_values, interaction_terms, 
                             title = 'Sobol indices', truncation_pct = 0.95, 
                             filename = filename, show = True,
                             fignum = 1, rv = r'\xi' )
    
from serialization_cpp import save_pce, load_pce
def test_oscillator():
    model = construct_data()
    pce = build_pce_from_file_data( 'pts.txt', 'vals.txt', model.rv_trans )
    filename = 'oscillator-pce.dat'
    save_pce( pce, filename, 1 )
    x = numpy.ones( (6,1 ) )
    print pce.evaluate_set( x )
    pce1 = PolynomialChaosExpansion()
    load_pce( pce1, filename, 1 )
    print pce1.evaluate_set( x )
    print pce1.dimension(), pce.dimension()
    compute_pce_rmse_error( pce, model, model.rv_trans )
    me, te, ie = get_sensitivities( pce )
    print te
    #plot_sensitivities( pce )



def sacramento_study():
    ranges = numpy.array( [65.2756244828626, 145.322542259761,
                           18.0894710898541, 149.999515390873,
                           0.264777513553855, 0.499996826528404,
                           0.018105653117895, 0.040667149144647,
                           3.66339053857432e-06, 0.05646498601088,
                           1.21504243924963, 180.013348512454,
                           0.675890033338491, 4.99999467330741,
                           1.0001803665577, 47.0439606200003,
                           41.7723522061654, 363.287950444946,
                           280.097478664303, 999.938531288531,
                           0.142072666489677, 0.24999923167236,
                           0.00588855817337181, 0.0106523425407948,
                           5.33344391756537e-07, 0.599985546539209] )
    rv_trans = RandomVariableTransformation()
    dist_types = ['uniform']*13
    means = numpy.zeros( ( 13 ), numpy.double )    # dummy for uniform
    std_devs = numpy.zeros( ( 13 ), numpy.double ) # dummy for uniform
    rv_trans.set_random_variables( dist_types, ranges, means, std_devs )
    
    
    pce = build_pce_from_file_data( 'sacramento/pts.txt', 
                                    'sacramento/vals.txt', rv_trans )


    
    pts = numpy.loadtxt( 'sacramento/pts.txt', delimiter = ',' )
    vals = numpy.loadtxt(  'sacramento/vals.txt', delimiter = ',' )
    vals = vals.reshape( vals.shape[0], 1 )

    #I = numpy.random.RandomState(seed).permutation( pts.shape[1] )[:num_subsamples]
    #pts = pts[:,I]
    #vals = vals[I,:]    
    #compute_pce_rmse_error_from_data( pce, pts, vals )
    me, te, ie = get_sensitivities( pce )
    print te

    

if __name__ == "__main__":
    seed = 3
    #num_subsamples = 3000
    #sacramento_study()
    test_oscillator()
    #test_sobol_function_sensitivities()
    #test_ishigami_sensitivities()
    
