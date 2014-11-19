


# dis = final_flopy_api.dis(mf, nlay, nrow, ncol, bounding_box, 
#                         nper, perlen, nstp, steady,)

# nwt = flopy.modflow.mfnwt.ModflowNwt(mf)

# bas = final_flopy_api.bas()
# # strt = dis.package.top.array - 10
# strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
# bas.perturb( np.random.random(4), mf, strt, nrow, ncol, bounding_box)

# delr = float(dis.package.delr.array[0])
# delc = float(dis.package.delc.array[0])
# n_var_upw = 3
# upw = final_flopy_api.upw(nlay, nrow, ncol, delr, delc, bounding_box, n_var_upw)
# upw.perturb(np.random.random(n_var_upw), mf)

# top = dis.package.top.array
# hk = upw.package.hk.array[0]
# riv = final_flopy_api.riv(nlay, nrow, ncol, bounding_box)
# riv.perturb(np.random.random(nper), mf,  top, hk, nper)

# wel = final_flopy_api.wel(nlay, nrow, ncol, bounding_box)
# wel.perturb(np.random.random(nper), mf, nper)





# bas
# ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
# ibound[:, :, 0] = -1
# ibound[:, :, -1] = -1


# ghb
# stageleft = [100.,10.]
# stageright = [10, 0.]

# wel
# pumping_rate = -200.

# dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc,
#                                    top=ztop, botm=botm[1:],
#                                    nper=nper, perlen=perlen, nstp=nstp, 
#                                    steady=steady)

# bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

#upw
# hk = 1+np.random.random((nlay,nrow,ncol))
# vka = 1 #np.ones((nlay,nrow,ncol))
# sy = 0.1
# ss = 1.e-4

# # upw and nwt replace lpf and pcg respectively 
# upw = flopy.modflow.ModflowUpw(mf,
#                 iupwcb = 0,
#                 # TODO this is rounded to -1e4 which affects results
#                 hdry = -9999,
#                 iphdry = 1, #hdryopt
#                 laytyp = 1, 
#                 layavg = 1, 
#                 chani = 1.0, 
#                 layvka = 1, 
#                 laywet = 0,
#                 hk = hk,
#                 vka = vka,
#                 sy=sy, 
#                 ss=ss
#                 )



# specified flux
# rch
# wel_sp1 = [[1, nrow/2, ncol/2, 0.]]
# wel_sp2 = [[1, nrow/2, ncol/2, 0.]]
# wel_sp3 = [[1, nrow/2, ncol/2, pumping_rate]]
# welllist = [wel_sp1, wel_sp2, wel_sp3]
# wel = flopy.modflow.ModflowWel(mf, layer_row_column_Q=welllist)

# # head dependent flux
# # riv
# # ghb
# def cond(stage, lay, col, row):
#     return hk[lay,col,row] * (stage - zbot) * delc

# #make list for stress period 1
# bound_sp1 = []
# for il in xrange(nlay):
#     for ir in xrange(nrow):
#         bound_sp1.append([il + 1, ir + 1, 0 + 1, stageleft[0], cond(stageleft[0],il,ir,0)])
#         bound_sp1.append([il + 1, ir + 1, ncol - 1, stageright[0], cond(stageright[0],il,ir,ncol-1)])
# print 'Adding ', len(bound_sp1), 'GHBs for stress period 1.'

# #make list for stress period 2
# bound_sp2 = []
# for il in xrange(nlay):
#     for ir in xrange(nrow):
#         bound_sp2.append([il + 1, ir + 1, 0 + 1, stageleft[1], cond(stageleft[1],il,ir,0)])
#         bound_sp2.append([il + 1, ir + 1, ncol - 1, stageright[1], cond(stageright[1],il,ir,ncol-1)])
# print 'Adding ', len(bound_sp2), 'GHBs for stress period 2.'

# #We do not need to make a list for stress period 3.
# #Flopy will automatically take the list and apply it
# #to the end of the simulation, if necessary
# boundlist = [bound_sp1, bound_sp2]

#Create the flopy ghb object
# ghb = flopy.modflow.ModflowGhb(mf, layer_row_column_head_cond=boundlist)
