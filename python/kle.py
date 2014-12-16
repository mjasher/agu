import os
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from correlation import correlation

# from fixed_parameters import working_directory, nrow, ncol, nlay, Lx, Ly, KLE_L, KLE_sigma, KLE_num_eig, KLE_mu, KLE_scale

# TODO check kle shouldn't take delr, delc rather than Lx Ly

# correlation kernel/function parameters
# KLE_sigma = 0.2 # variance or sill
# KLE_L = delr # correlation length

# KLE_num_eig = num_dims  #1157.44467688 sec
# KLE_mu = 3*np.ones(nlay*nrow*ncol)
# KLE_scale = 1

working_directory = os.getcwd()+'/'

class kle():
	def __init__(self, KLE_sigma, KLE_L,
				KLE_num_eig,
				KLE_mu, KLE_scale,
				nrow, ncol, nlay, Lx, Ly):

		self.KLE_mu = KLE_mu
		self.KLE_scale = KLE_scale
		self.nrow =	nrow
		self.ncol =	ncol
		self.nlay =	nlay

		# use existing KLE if dimensions match
# TODO remove False
		# if False: 
		if os.path.exists(working_directory+'kle.npz'):
			saved_kle = np.load(working_directory+'kle.npz')
			if saved_kle['KLE_L']==KLE_L and saved_kle['KLE_sigma']==KLE_sigma and saved_kle['KLE_num_eig']==KLE_num_eig and saved_kle['nrow']==nrow and saved_kle['ncol']==ncol and saved_kle['Lx']==Lx and saved_kle['Ly']==Ly:
				self.eig_vals=saved_kle['eig_vals']
				self.eig_vecs=saved_kle['eig_vecs']
			else:
				self.eig_vals, self.eig_vecs = self.make_kle(KLE_sigma, KLE_L, KLE_num_eig, nrow, ncol, nlay, Lx, Ly)
		else:
			self.eig_vals, self.eig_vecs = self.make_kle(KLE_sigma, KLE_L, KLE_num_eig, nrow, ncol, nlay, Lx, Ly)



	def make_kle(self, KLE_sigma, KLE_L,
				KLE_num_eig,
				nrow, ncol, nlay, Lx, Ly):

		print "making KLE, this will take a long time for large dimensions"

		print KLE_sigma, KLE_L, KLE_num_eig, nrow, ncol, nlay, Lx, Ly


		# initialise swig wrapped cpp class

		C_matrix = correlation.C_matrix(KLE_sigma, KLE_L)
		C_matrix.set_dims(nrow, ncol, Lx, Ly)

		out_vec = np.zeros((nrow*ncol), 'd')
		def KLE_Av(v):
			C_matrix.av_no_C(nrow*ncol, v, out_vec)
			return out_vec

		KLE_A = LinearOperator((nrow*ncol,nrow*ncol), matvec=KLE_Av, dtype='d')

		t1 = time.time()
		eig_vals, eig_vecs = eigsh( KLE_A, k=KLE_num_eig)
		t2 = time.time()


		# sometimes one gets -v rather than v?
		for i in range(KLE_num_eig):
			print "NORM", np.linalg.norm(eig_vecs[:,i])
			assert np.allclose(KLE_A.matvec(eig_vecs[:,i]), eig_vals[i]*eig_vecs[:,i])

		print "=================================="
		print "SVD took ", t2-t1, "seconds and "
		print "they seem to indeed be eigen vectors"
		print "=================================="


# plot eigenvectors
		from mpl_toolkits.mplot3d import axes3d
		import matplotlib.pyplot as plt
		fig = plt.figure(figsize=plt.figaspect(0.2))
		for i in range(eig_vecs.shape[1]):
			ax = fig.add_subplot(1, eig_vecs.shape[1], i, projection='3d')
			x = np.arange(0, nrow*Lx, Lx)
			X = np.empty((nrow, ncol))
			for col in range(ncol):
				X[:,col] = x
			y = np.arange(0, ncol*Ly, Ly)
			Y = np.empty((nrow, ncol))
			for row in range(nrow):
				Y[row,:] = y
			Z = eig_vecs[:,i].reshape((nrow, ncol))
			ax.plot_wireframe(X, Y, Z)
		plt.show()

# plot eigenvals

		# import matplotlib.pyplot as plt
		plt.plot(eig_vals, 'o-')
		plt.show()

		print "eig_vals", eig_vals

		np.savez(working_directory+'kle.npz', eig_vals=eig_vals, eig_vecs=eig_vecs, KLE_L=KLE_L, KLE_sigma=KLE_sigma, KLE_num_eig=KLE_num_eig, nrow=nrow, ncol=ncol, Lx=Lx, Ly=Ly)

		return eig_vals, eig_vecs



	def compute(self, modes):
		# print "eig vals", self.eig_vals
		coefs = np.sqrt(self.eig_vals) * modes # TODO this 3 is arbirtrary # elementwise
		truncated_M = self.KLE_mu + self.KLE_scale * np.dot( self.eig_vecs, coefs)
		return truncated_M.reshape(self.nlay, self.nrow, self.ncol)
