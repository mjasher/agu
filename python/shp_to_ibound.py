import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np

from shapely.geometry import mapping, shape
from shapely.ops import cascaded_union
import shapely # wrapper for libgeos
from osgeo import ogr, osr, gdal

# easier but less flexible
# import shapefile
# import fiona # wrapper for ogr

class shp_to_ibound():

	# def __init__(self):
	# 	pass

	# load and reproject shapefile, 
	#---------------------------------------
	def load_shp(self, nlay, nrow, ncol):
		driver = ogr.GetDriverByName('ESRI Shapefile')
		dataset = driver.Open("STE_2011_AUST/STE_2011_AUST.shp")
		# input Spatial Reference from Layer
		inLayer = dataset.GetLayer()
		inSpatialRef = inLayer.GetSpatialRef()

		# output SpatialReference, we want meters for modflow
		outSpatialRef = osr.SpatialReference()
		outSpatialRef.ImportFromEPSG(3395)

		# create the CoordinateTransformation
		transform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

		feat = inLayer.GetFeature(0)	# nsw,vic,qld,sa,wa
		geom = feat.GetGeometryRef()
		ring = geom.GetGeometryRef(0)
		ring.Transform(transform)

		# points = np.array(geom.GetGeometryRef(0).GetPoints())
		# extent = [np.min(points[:,0]), np.min(points[:,1]), np.max(points[:,0]), np.max(points[:,1]) ], 
		extent = geom.GetEnvelope()
		Ly = extent[3]-extent[2]
		Lx = extent[1]-extent[0]

		# make an empty row/col on each side so perturbations are within domain
		bLx = Lx*ncol/(ncol-2)
		bLy = Ly*nrow/(nrow-2)
		
		delr = bLx / ncol
		delc = bLy / nrow

		self.feat = feat.Clone()
		self.ring = ring.Clone()
		self.delr = delr
		self.delc = delc

		# expand east delc and north delr for empty
		x_min, x_max, y_min, y_max = ring.GetEnvelope()
		
		# empty row/col
		self.extent = (x_min - delr*1.00000001, x_max + delr*1.00000001*1.00000001, y_min - delc*1.00000001, y_max + delc*1.00000001)
		
		# self.ring = ogr.Geometry(ogr.wkbLinearRing)
		# for p in range(ring.GetPointCount()):
		# 	point = ring.GetPoint(p)
		# 	# newRing.AddPoint(point[0], point[1])
		# 	self.ring.AddPoint(point[0]+ delc, point[1]+ delr)
		# self.ring.CloseRings()

		# return Lx, Ly
		return bLx, bLy, ring.GetPointCount()


	# perturb
	#---------------------------------------
	def perturbed(self, perturbations):
		feat = self.feat
		ring = self.ring
		delr = self.delr
		delc = self.delc
		x_min, x_max, y_min, y_max = self.extent

		# Create a new perturbed polygon
		newRing = ogr.Geometry(ogr.wkbLinearRing)

		# scale = 1e5 # meter
		for p in range(ring.GetPointCount()):
			point = ring.GetPoint(p)
			# newRing.AddPoint(point[0], point[1])
			# newRing.AddPoint(point[0]+ scale*(np.random.random() - 0.5), point[1]+ scale*(np.random.random() - 0.5))
			newRing.AddPoint(point[0]+ delr*2*(perturbations[p,0] - 0.5), point[1]+ delc*2*(perturbations[p,1] - 0.5))
		# newRing.CloseRings()
		first = newRing.GetPoint(0)
		newRing.AddPoint(first[0], first[1]) # close linearring

		# Create a new polygon
		poly = ogr.Geometry(ogr.wkbPolygon)
		poly.AddGeometry(newRing)
		poly.AssignSpatialReference(ring.GetSpatialReference())


		# fix self-intersections (resulting from perturbations)
		# print 'valid', poly.IsValid()
		# poly = poly.Buffer(0.0) # often deletes shape
		# print 'now valid', poly.IsValid()

		newFeat = feat.Clone()
		newFeat.SetGeometry(poly)

		# convert vector layer to array 
		#---------------------------------------
		# create a new layer with only the given feature
		driver = ogr.GetDriverByName('Memory')
		datasource = driver.CreateDataSource('fileName')
		newLayer = datasource.CreateLayer('layerName',geom_type=ogr.wkbPolygon)
		newLayer.CreateFeature( newFeat )

		# Define pixel_size and NoData value of new raster
		# pixel_size = 50000
		NoData_value = 255

		# Open the datasource and read in the extent
		# source_ds = ogr.Open(vector_fn)
		# source_srs = newLayer.GetSpatialRef()
		# print "source srs", source_srs
		# x_min, x_max, y_min, y_max = newLayer.GetExtent()

		# TODO create padding here
		# TODO perturb boundary near a given point

		# Create the destination data source
		x_res = int((x_max - x_min) / delr)
		y_res = int((y_max - y_min) / delc)
		# x_res = int((x_max - x_min) / pixel_size)
		# y_res = int((y_max - y_min) / pixel_size)
		target_ds = gdal.GetDriverByName('MEM').Create('', x_res, y_res, gdal.GDT_Byte)
		target_ds.SetGeoTransform((x_min, delr, 0, y_max, 0, -delc))
		# target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
		band = target_ds.GetRasterBand(1)
		band.SetNoDataValue(NoData_value)

		# Rasterize
		gdal.RasterizeLayer(target_ds, [1], newLayer, burn_values=[1])

		# Read as array
		array = band.ReadAsArray()

		array = np.array(array, 'i')

		array[array == 0] = 0
		return array

		# plot
		#---------------------------------------
		# points = np.array(poly.GetGeometryRef(0).GetPoints())
		# fig = plt.figure()
		# plt.fill(points[:,0], points[:,1] )
		# plt.savefig('shp_to_ibound.png')
		# plt.show()



if __name__ == '__main__':
	this = shp_to_ibound()
	Lx,Ly, points = this.load_shp(1,18,17)
	print "Lx, Ly", Lx, Ly, points
	# print this.perturbed(np.random.random((points,2)))
	print this.perturbed(0.5*np.ones((points,2)))
