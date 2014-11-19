import flopy
import numpy as np
# np.random.seed(987654321) # keep things consistent for testing

import final_flopy_api

"""
title: Parameterizing a conceptual groundwater model for sensitivity analysis. From reading common data types (eg. shapefiles) to sensitivity analysis using a Polynomial Chaos surrogate.
author: michael.james.asher@gmail.com
date: November 2014
"""

"""
[Flopy](code.google.com/p/flopy) is a python wrapper for the groundwater modelling software MODFLOW (water.usgs.gov/ogw/modflow/MODFLOW.html). Using flopy, this file contains classes for each MODFLOW input package with "initialize" and "perturb" methods. The "initialize" method reads in data and sets fixed and default parameters. "Perturb" is intended to take whatever inputs one wishes to perform sensitivity analysis for.

Flopy isn't designed to have packages changed individually (eg. subsequent calls of ModflowDis will not change (nlay,nrow,ncol)). Perhaps simply writing functions which took all necessary inputs for each package and produced just that input file would be more flexible.
"""

working_directory = '/home/mikey/Dropbox/pce/agu/python'
# MODFLOW_executable = '/home/mikey/Dropbox/raijin/Unix/src/mf2005'
MODFLOW_NWT_executable = '/home/mikey/Dropbox/raijin/MODFLOW-NWT_1.0.9/MF_NWT'
modelname = 'flopy_model'

bounding_box = [[-30.82,150.54],[-31.04,151.01]] #  top left, bottom right

nlay,nrow,ncol = (1,10,20)
nper = 3
perlen = [1, 100, 100]
nstp = [1, 100, 100]
steady = [True, False, False]


wells = [
    [ 150.74673843279015, -30.885048778979645], 
    [ 150.87479782523587, -30.876906574734466],
    [ 150.91599655570462, -30.987047586140907],
    [ 150.69695663871244, -31.005293991028097]
]

x_min = bounding_box[0][1]
x_max = bounding_box[1][1]
y_min = bounding_box[1][0]
y_max = bounding_box[0][0]

pixelWidth = (x_max - x_min)/ncol
pixelHeight = (y_max - y_min)/nrow

interesting_points = []
for well in wells:
    interesting_points.append([0, int((well[1]-y_min)/pixelHeight), int((well[0]-x_min)/pixelWidth) ])








num_dims = 1
num_outputs = 4



# ---------------
# function used by PCE
# ---------------

def f( x ):

    mf = flopy.modflow.Modflow(modelname, 
                            exe_name = MODFLOW_NWT_executable,
                            model_ws = working_directory+modelname+'-mf/',
                            # verbose = False,
                            # silent = 1,
                            )

    # TODO top is just 0, botm=-50, fix to match topo or use NGIS lithology logs
    dis = final_flopy_api.dis(mf, nlay, nrow, ncol, bounding_box, 
                            nper, perlen, nstp, steady,)

    delr = float(dis.package.delr.array[0])
    delc = float(dis.package.delc.array[0])

    nwt = flopy.modflow.mfnwt.ModflowNwt(mf)

    words = ['head','drawdown','budget', 'phead', 'pbudget']
    save_head_every = 1
    oc = flopy.modflow.ModflowOc(mf, words=words, save_head_every=save_head_every)

    bas = final_flopy_api.bas()


    print "x", x
    par = np.ones(14)
    par[9] = x[0]

    #TODO move inits out of f
    strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
    strt[:, :, 0] = 10.
    strt[:, :, -1] = 0.
    # move_boundary = np.random.random(4)
    move_boundary = par[0:4]
    bas_perturb = bas.perturb( mf, strt, nrow, ncol, bounding_box, move_boundary = move_boundary)
    bas_perturb.write_file()

    n_var_upw = 3
    upw = final_flopy_api.upw(nlay, nrow, ncol, delr, delc, bounding_box, n_var_upw)
    # kle_modes = np.random.random(n_var_upw)
    kle_modes = par[4:7]
    upw_perturb = upw.perturb(mf, kle_modes = kle_modes)
    upw_perturb.write_file()

    wel = final_flopy_api.wel(nlay, nrow, ncol, bounding_box)
    wel_rate = -100*par[7:10]
    # wel_rate = -100*np.random.random(nper)
    # wel_rate  =[0,-100,-500]
    # print "wel_rate", wel_rate
    wel.perturb(mf, nper, rates = wel_rate).write_file()

    hk = upw_perturb.hk.array 
    top = dis.package.top.array
    ibound = bas_perturb.ibound # can't have boundary conditions where ibound <= 0
    riv = final_flopy_api.riv(nlay, nrow, ncol, bounding_box, ibound)
    stage_height = par[10:13]
    # stage_height = np.random.random(nper)
    # stage_height = [0, 2, 2]
    # print "stage_height", stage_height
    riv.perturb(mf, top, hk[0], nper, stage_height = stage_height).write_file()

    rch = final_flopy_api.rch(nlay, nrow, ncol, bounding_box)
    # intensity = 0.5*np.random.random()
    intensity = 0.2*par[13]
    rch.perturb(mf, intensity).write_file()

    mf.write_input()
    #run the model
    success, mfoutput = mf.run_model2(
                                # silent=True, pause=False
                                )
    if not success:
        raise Exception('MODFLOW did not terminate normally.')

    headobj = flopy.utils.binaryfile.HeadFile(working_directory+modelname+'-mf/'+modelname+'.hds')

    heads = headobj.get_data(totim=201)
    y = np.array([ heads[p[0],p[1],p[2]] for p in interesting_points ])
    print "y", y
    return y

    # to plot
    # return headobj




# todo could generate contours using osgeo.gdal.ContourGenerate

def results_to_geojson(headobj):

    x_min = bounding_box[0][1]
    x_max = bounding_box[1][1]
    y_min = bounding_box[1][0]
    y_max = bounding_box[0][0]

    pixelWidth = (x_max - x_min)/ncol
    pixelHeight = (y_max - y_min)/nrow

    outputs = []

    mytimes = [1.0, 101.0, 201.0]
    for iplot, time in enumerate(mytimes):
        print '*****Processing time: ', time
        head = headobj.get_data(totim=time)
        outputs.append({
            "time": time,
            "bottomLeft": { "lat": y_min, "lng": x_min },
            "pixelHeight": pixelHeight,
            "pixelWidth": pixelWidth,
            "array": head[0].tolist() 
        })
        print 'Head statistics'
        print '  min: ', head.min()
        print '  max: ', head.max()
        print '  std: ', head.std()


    import json
    with open('/home/mikey/Dropbox/pce/interactions/agu/data/outputs.json', 'w') as f:
        f.write(json.dumps(outputs))

# results_to_geojson()







def plot_results(headobj):

    import matplotlib.pyplot as plt
    Lx = ncol*delr
    Ly = nrow*delc

    #Create the headfile object
    times = headobj.get_times()

    #Setup contour parameters
    levels = np.arange(1, 10, 1)
    extent = (delr/2., Lx - delr/2., delc/2., Ly - delc/2.)
    print 'Levels: ', levels
    print 'Extent: ', extent

    mytimes = [1.0, 101.0, 201.0]
    for iplot, time in enumerate(mytimes):
        print '*****Processing time: ', time
        head = headobj.get_data(totim=time)
        #Print statistics
        print 'Head statistics'
        print '  min: ', head.min()
        print '  max: ', head.max()
        print '  std: ', head.std()

        #Create the plot
        #plt.subplot(1, len(mytimes), iplot + 1, aspect='equal')
        plt.subplot(1, 1, 1, aspect='equal')
        plt.title('stress period ' + str(iplot + 1))
        plt.imshow(head[0, :, :], extent=extent, cmap='BrBG', vmin=0., vmax=10.)
        plt.colorbar()
        CS = plt.contour(head[0, :, :], levels=levels, extent=extent)
        plt.clabel(CS, inline=1, fontsize=10, fmt='%1.1f')
        plt.show()

    plt.show()


# print head.get_data(totim=201)[0, :, :]
# plot_results(headobj)

if __name__ == '__main__':
    # headobj = f(np.random.random(1))
    # results_to_geojson(headobj)

    # TODO package doesnn't updata
    print f([0.616718767492])
    print f([0.216718767492])

