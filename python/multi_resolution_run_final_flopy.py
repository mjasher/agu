import flopy
import numpy as np
# np.random.seed(987654321) # keep things consistent for testing
import time
import final_flopy_api

"""
title: Parameterizing a conceptual groundwater model for sensitivity analysis. From reading common data types (eg. shapefiles) to sensitivity analysis using a Polynomial Chaos surrogate.
author: michael.james.asher@gmail.com
date: November 2014
"""

"""
[Flopy](code.google.com/p/flopy) is a python wrapper for the groundwater modelling software MODFLOW (water.usgs.gov/ogw/modflow/MODFLOW.html). Using flopy, this file contains functions for each MODFLOW input package with "initialize" and "perturb" methods. The "initialize" method reads in data and sets fixed and default parameters. "Perturb" is intended to take whatever inputs one wishes to perform sensitivity analysis for.

Flopy isn't designed to have packages changed individually (eg. subsequent calls of ModflowDis will not change (nlay,nrow,ncol)). Perhaps simply writing functions which took all necessary inputs for each package and produced just that input file would be more flexible.
"""

working_directory = '/home/mikey/Dropbox/pce/agu/python/'
# MODFLOW_executable = '/home/mikey/Dropbox/raijin/Unix/src/mf2005'
MODFLOW_NWT_executable = '/home/mikey/Dropbox/raijin/MODFLOW-NWT_1.0.9/MF_NWT'
modelname = 'flopy_model'

bounding_box = [[-30.82,150.54],[-31.04,151.01]] #  top left, bottom right

# nlay,nrow,ncol = (1,10,20)
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

import json
data_dir = '/home/mikey/Dropbox/pce/agu/data/'
with open(data_dir+'monitoring_wells.json', 'w') as f:
    f.write(json.dumps( { "type": "FeatureCollection",
    "features": [
      { "type": "Feature",
        "geometry": {"type": "Point", "coordinates": p},
        "properties": {"index": i},
        } for i,p in enumerate(wells) ]
     } ))


x_min = bounding_box[0][1]
x_max = bounding_box[1][1]
y_min = bounding_box[1][0]
y_max = bounding_box[0][0]

num_dims = 10
# num_dims = 14
num_outputs = 4

p_names = [
            # "well location 1", "well location 2", "well location 3", "well location 4",
            "hk KLE mode 1","hk KLE mode 2","hk KLE mode 3",
            "well rate 1","well rate 2","well rate 3",
            "stage height 1","stage height 2","stage height 3",
            "rch intensity"
            ]


# model = mf_model

class mf_model():
    def __init__(self,nlay,nrow,ncol):

        self.nlay = nlay
        self.nrow = nrow
        self.ncol = ncol

        pixelWidth = (x_max - x_min)/ncol
        pixelHeight = (y_max - y_min)/nrow

        # locations for output heads
        self.interesting_points = []
        for well in wells:
            self.interesting_points.append([0, int((well[1]-y_min)/pixelHeight), int((well[0]-x_min)/pixelWidth) ])


        # TODO how to cope with errors 

        self.dis_init = final_flopy_api.dis_init(nlay, nrow, ncol, bounding_box)

        (top, botm, delr, delc) = self.dis_init

        self.bas_init = final_flopy_api.bas_init()

        n_var_upw = 3
        self.upw_init = final_flopy_api.upw_init(nlay, nrow, ncol, delr, delc, bounding_box, n_var_upw)

        self.wel_init = final_flopy_api.wel_init(nlay, nrow, ncol, bounding_box)

        self.rch_init = final_flopy_api.rch_init(nlay, nrow, ncol, bounding_box)

        self.riv_init = final_flopy_api.riv_init(nlay, nrow, ncol, bounding_box)



    # ---------------
    # function used by PCE
    # ---------------
    def f(self, x ):

        times = []

        nlay = self.nlay
        nrow = self.nrow
        ncol = self.ncol


        print "x", x
        par = x
        # par = np.array([0.60412155, 0.85414241, 0.18801196, 0.75726005, 0.62560195, 0.55517539 , 0.30901852, 0.30107194, 0.84115518, 0.88161805, 0.53778622, 0.95330613 , 0.10741349, 0.09261109])

        mf = flopy.modflow.Modflow(modelname, 
                            exe_name = MODFLOW_NWT_executable,
                            model_ws = working_directory+modelname+'-mf/',
                            # verbose = False,
                            # silent = 1,
                            )

        nwt = flopy.modflow.mfnwt.ModflowNwt(mf)

        # TODO top is just 0, botm=-50, fix to match topo or use NGIS lithology logs
        dis_perturb = final_flopy_api.dis_perturb(self.dis_init, mf, nlay, nrow, ncol, nper, perlen, nstp, steady)
        # delr = float(dis_perturb.delr.array[0])
        # delc = float(dis_perturb.delc.array[0])

        strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
        strt[:, :, 0] = 10.
        strt[:, :, -1] = 0.
        # move_boundary = 1.0*par[0:4]
        move_boundary = np.zeros((4))
        bas_perturb = final_flopy_api.bas_perturb(self.bas_init, mf, strt, nrow, ncol, bounding_box, move_boundary = move_boundary)
        bas_perturb.write_file()

        # kle_modes = par[4:7]
        kle_modes = par[0:3]
        # assert len(kle_modes) ==  n_var_upw
        # kle_modes = np.ones((3))
        upw_perturb = final_flopy_api.upw_perturb(self.upw_init, mf, kle_modes = kle_modes)
        upw_perturb.write_file()
        # print upw_perturb.hk.array

        wel_rate = -100*par[3:6]
        assert len(wel_rate) == nper
        # move_boundary = 2*par[10:]
        # move_boundary = np.zeros(4)
        wel_perturb = final_flopy_api.wel_perturb(self.wel_init, mf, nper, rates = wel_rate, move_boundary = move_boundary)
        wel_perturb.write_file()

        hk = upw_perturb.hk.array 
        top = dis_perturb.top.array
        ibound = bas_perturb.ibound # can't have boundary conditions where ibound <= 0
        # print "ibound", (ibound==original_ibound)[0][2:3]
        self.riv_init = final_flopy_api.riv_init(nlay, nrow, ncol, bounding_box)
        stage_height = par[6:9]
        stage_height[0] = stage_height[0]*0.25
        assert len(stage_height) == nper
        riv_perturb = final_flopy_api.riv_perturb(self.riv_init, mf, top, hk[0], nper, stage_height = stage_height, ibound=ibound)
        riv_perturb.write_file()

        intensity = 0.2*par[9]
        rch_perturb =  final_flopy_api.rch_perturb(self.rch_init, mf, intensity)
        rch_perturb.write_file()

        words = ['head','drawdown','budget', 'phead', 'pbudget']
        save_head_every = 1
        oc = flopy.modflow.ModflowOc(mf, words=words, save_head_every=save_head_every)

        mf.write_input()
        #run the model
        t0 = time.time()
        success, mfoutput = mf.run_model2(
                                    # silent=True, pause=False
                                    )
        t1 = time.time()
        times.append(t1-t0)
        if not success:
            raise Exception('MODFLOW did not terminate normally.')

        headobj = flopy.utils.binaryfile.HeadFile(working_directory+modelname+'-mf/'+modelname+'.hds')

        # plot
        # final_flopy_api.results_to_geojson(headobj, nrow, ncol, bounding_box)

        heads = headobj.get_data(totim=201)
        
        y = np.array([ heads[p[0],p[1],p[2]] for p in self.interesting_points ], dtype='d')
        # if np.isnan(y[0]):
        #     y = np.zeros(num_outputs)
        print "y", y
        return y

        # to plot
        # return headobj



    # results_to_geojson()




if __name__ == '__main__':
    this_model = mf_model(1,10,20)
    print this_model.f(np.random.random(num_dims))