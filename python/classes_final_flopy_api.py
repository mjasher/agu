import flopy
import numpy as np
# np.random.seed(987654321) # keep things consistent for testing

from osgeo import ogr, osr, gdal

import json
import os

import kle
import ogr_to_modflow

"""
title: Parameterizing a conceptual groundwater model for sensitivity analysis. From reading common data types (eg. shapefiles) to sensitivity analysis using a Polynomial Chaos surrogate.
author: michael.james.asher@gmail.com
date: November 2014
"""


'''
It's a big file, only contains one class for each modflow package.

# TODO

* go through below packages and replace with mix of flopy_api.py and ogr_to_modflow.py
* run the model to get it working
* wrap in PC
* produce output data and matplotlib
* make shmancy d3js plots

* note I name what should be delr,delc as Lx,Ly in my kle module, ensure it's given delr, delc 

 '''
		
data_dir = '/home/mikey/Dropbox/pce/agu/data/'

plot = False


"""
dis 
"""

class dis():
	def __init__(self, mf, nlay, nrow, ncol, bounding_box,
		nper, perlen, nstp, steady):

		
		ogr_to_modflow.build_grid(data_dir+"APPT250K_WatercourseLines_line.json", data_dir+'grid.json',
			nrow, ncol)

		# rasterize contours

		def rasterize_top(source_layer, clipping_poly):
			centroid = clipping_poly.Centroid()

			properties = [
				(feature.GetGeometryRef().Distance(centroid), feature.GetField('ELEVATION'))
				for feature in source_layer
			]
			sorted_properties = np.sort(
				np.array(
					properties, 
					dtype = [('distance', float), ('elevation', float)]
				), 
				order='distance'
			)
			return sorted_properties[0]['elevation']

		raster = ogr_to_modflow.ogr_to_modflow(
			data_dir+"APPT250K_Contours_line.json",
			nrow, ncol,
			rasterize_top,
			bounding_box
		) 
		
		if plot:
			with open(data_dir+"top.json", 'w') as f:
				f.write(json.dumps( raster ))

		# approximate cellwidth and height in meters

		p1 = ogr.Geometry(ogr.wkbPoint)
		p1.AddPoint(raster["bottomLeft"]["lng"], raster["bottomLeft"]["lat"])			
		p2 = ogr.Geometry(ogr.wkbPoint)
		p2.AddPoint(raster["bottomLeft"]["lng"]+raster["pixelWidth"], raster["bottomLeft"]["lat"])
		delc = ogr_to_modflow.projected_distance(4326, 3857, p1, p2)
		
		p1 = ogr.Geometry(ogr.wkbPoint)
		p1.AddPoint(raster["bottomLeft"]["lng"], raster["bottomLeft"]["lat"])	
		p2 = ogr.Geometry(ogr.wkbPoint)
		p2.AddPoint(raster["bottomLeft"]["lng"], raster["bottomLeft"]["lat"]+raster["pixelHeight"])		
		delr = ogr_to_modflow.projected_distance(4326, 3857, p1, p2)

		top = np.array(raster["array"])
		top = 0 # TODO
		botm = top - 50

		self.package = flopy.modflow.ModflowDis(mf, 
			nlay = nlay, nrow = nrow, ncol = ncol, 
			delr = delr, delc = delc,
			top = top, botm = botm,
			nper = nper, perlen = perlen, nstp = nstp,
			steady = steady)

"""
bas 
"""

def distance(a,b):
	return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def index_of_closest(ring, interesting_point):
	min_dist = distance(ring.GetPoint(0), interesting_point)
	min_p = 0
	for p in range(ring.GetPointCount()):
		point = ring.GetPoint(p)
		d = distance(point, interesting_point)
		if d < min_dist:
			min_dist = d
			min_p = p
	return min_p

class bas():
    def __init__(self):

		source_ds = ogr.Open(data_dir+'simplified_aquifer_boundary.json')
		source_layer = source_ds.GetLayer()

		feat = source_layer.GetFeature(0)
		geom = feat.GetGeometryRef()
		ring = geom.GetGeometryRef(0) # exterior ring
		
		interesting_points = [( 150.59327316703275, -30.867074106093657, 0), (150.62691879691556, -30.963685131712417, 0)]
		
		self.perturb_points = [ index_of_closest(ring, p) for  p in interesting_points ]

		self.feat = feat.Clone()
		self.ring = ring.Clone()

        # self.package = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    
    def perturb(self, mf, strt, nrow, ncol, bounding_box, move_boundary):
		
		feat = self.feat
		ring = self.ring
		perturb_points = self.perturb_points

		x_min = bounding_box[0][1]
		x_max = bounding_box[1][1]
		y_min = bounding_box[1][0]
		y_max = bounding_box[0][0]

		pixelWidth = (x_max - x_min)/ncol
		pixelHeight = (y_max - y_min)/nrow

		# create a new ring with perturbed points
		newRing = ogr.Geometry(ogr.wkbLinearRing)

		points = [ ring.GetPoint(i) for i in range(ring.GetPointCount()) ]

		for i,p in enumerate(perturb_points):
			points[p] = (points[p][0]+ pixelWidth*2*(move_boundary[0+i*2] - 0.5),
				 		points[p][1]+ pixelHeight*2*(move_boundary[1+i*2] - 0.5))

		for p in points:
			newRing.AddPoint(*p)
		first = newRing.GetPoint(0)
		newRing.AddPoint(first[0], first[1]) # close linearring

		geom = ogr.Geometry(ogr.wkbPolygon)
		geom.AddGeometry(newRing)


		# add other rings from original geom
		old_geom = feat.GetGeometryRef()
		for i in range(1,old_geom.GetGeometryCount()):
			geom.AddGeometry(old_geom.GetGeometryRef(i))


		# create geojson of perturbed to check
		if plot:
			if os.path.exists('/home/mikey/Dropbox/pce/agu/data/'+'perturbed_boundary.json'):
				os.remove('/home/mikey/Dropbox/pce/agu/data/'+'perturbed_boundary.json')
			geom.AssignSpatialReference(ring.GetSpatialReference())

			newFeat = feat.Clone()
			newFeat.SetGeometry(geom)

			driver = ogr.GetDriverByName('GeoJSON')
			datasource = driver.CreateDataSource('/home/mikey/Dropbox/pce/agu/data/'+'perturbed_boundary.json')
			newLayer = datasource.CreateLayer('layerName',geom_type=ogr.wkbPolygon)
			newLayer.CreateFeature( newFeat )
		# # 

		# rasterize

		array = np.empty((nrow,ncol))
		for row in range(nrow):
			for col in range(ncol):

				# Create clipping polygon
				ring = ogr.Geometry(ogr.wkbLinearRing)
				ring.AddPoint(x_min+(col)*pixelWidth, y_min+(row)*pixelHeight)
				ring.AddPoint(x_min+(col+1)*pixelWidth, y_min+(row)*pixelHeight)
				ring.AddPoint(x_min+(col+1)*pixelWidth, y_min+(row+1)*pixelHeight)
				ring.AddPoint(x_min+(col)*pixelWidth, y_min+(row+1)*pixelHeight)
				ring.AddPoint(x_min+(col)*pixelWidth, y_min+(row)*pixelHeight)
				clipping_poly = ogr.Geometry(ogr.wkbPolygon)
				clipping_poly.AddGeometry(ring)
				
				if geom.Intersection(clipping_poly).Area() == 0:
					array[row,col] = 0
				else:
					array[row,col] = 1

		if plot:
			with open(data_dir+"perturbed_ibound.json", 'w') as f:
				f.write(json.dumps( 
					{
						"bottomLeft": { "lat": y_min, "lng": x_min },
						"pixelHeight": pixelHeight,
						"pixelWidth": pixelWidth,
						"array": array.tolist() 
					}
				))
			
		ibound = array
		
		return flopy.modflow.ModflowBas(mf, ibound=ibound, hnoflo=0.0, strt=strt)

"""
upw 
"""
class upw():
	def __init__(self, nlay, nrow, ncol, delr, delc, bounding_box, num_dims_upw):
		
		def rasterize_hk(source_layer, clipping_poly):
			properties = np.array([
				[feature.GetGeometryRef().Clone().Intersection(clipping_poly).Area(),
					feature.GetField('HydKValue')]
				for feature in source_layer
			])
			return np.average(properties[:,1], weights=properties[:,0])

		self.hk_mean =  ogr_to_modflow.ogr_to_modflow(
				data_dir+"IGWWaterTableHydraulicConductivity.json",
				nrow, ncol,
				rasterize_hk,
				bounding_box) 
			
		if plot:
			with open(data_dir+"hk_mean.json", 'w') as f:
				f.write(json.dumps(self.hk_mean))

		def rasterize_sy(source_layer, clipping_poly):
			properties = np.array([
				[feature.GetGeometryRef().Clone().Intersection(clipping_poly).Area(),
					feature.GetField('SpecYield')]
				for feature in source_layer
				if feature.GetField('SpecYield') != None
			])
			if np.sum(properties[:,0]) == 0:
				return -9999
			else:
				return np.average(properties[:,1], weights=properties[:,0])

		self.sy_mean = ogr_to_modflow.ogr_to_modflow(
			data_dir+"IGWWaterTableHydraulicConductivity.json",
			nrow, ncol,
			rasterize_sy,
			bounding_box
		) 
		
		if plot:	
			with open(data_dir+"sy_mean.json", 'w') as f:
				f.write(json.dumps(self.sy_mean ))

		self.upw_kle = kle.kle(KLE_sigma = 0.2, 
			KLE_L = delr,
			KLE_num_eig = num_dims_upw,
			KLE_mu = 3*np.ones(nlay*nrow*ncol),
			KLE_scale = 1,
			nrow = nrow, ncol = ncol, nlay = nlay, Lx = delc, Ly = delr)

	def perturb(self, mf, kle_modes):
		
		upw_kle = self.upw_kle

		iupwcb = 0
		hdry = -9999 # TODO this is rounded to -1e4 which affects results
		iphdry = 1 #hdryopt
		laytyp = 1 
		layavg = 1 
		chani = 1.0 
		layvka = 1 
		laywet = 0
		hk = self.hk_mean["array"] + upw_kle.compute(kle_modes)
		# hk = 1+np.random.random((nlay,nrow,ncol)) # default
		vka = 1 #np.ones((nlay,nrow,ncol))
		sy = np.array(self.sy_mean["array"]) #0.1
		ss = 1.e-4 # np.array(self.sy_mean["array"])*1e-4 #1.e-4


		return flopy.modflow.ModflowUpw(mf,
			iupwcb = iupwcb, 
			hdry = hdry, 
			iphdry = iphdry,  #hdryopt
			laytyp = laytyp,  
			layavg = layavg,  
			chani = chani, 
			layvka = layvka,  
			laywet = laywet, 
			hk = hk,
			vka = vka,
			sy = sy, 
			ss = ss
			)


"""
riv 
"""
class riv():
    def __init__(self, nlay, nrow, ncol, bounding_box, ibound):

		# transform so we can use length of river in meters
		inDs = ogr.Open(data_dir+"APPT250K_WatercourseLines_line.json")
		inLayer = inDs.GetLayer()
		inSpatialRef = inLayer.GetSpatialRef()

		# output SpatialReference, we want meters for modflow
		outSpatialRef = osr.SpatialReference()
		outSpatialRef.ImportFromEPSG(3857)

		# create the CoordinateTransformation
		transform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)


		def rasterize_riv(source_layer, clipping_poly):
			properties = []
			for feature in source_layer:
				if feature.GetField('HIERARCHY') == "Major":	
					clipped = clipping_poly.Clone().Intersection(feature.GetGeometryRef())
					# clipped = feature.GetGeometryRef().Clone().Intersection(clipping_poly)
					clipped.Transform(transform)
					properties.append(clipped.Length())
			return np.sum(properties)


		raster =  ogr_to_modflow.ogr_to_modflow(
			data_dir+"APPT250K_WatercourseLines_line.json",
			nrow, ncol,
			rasterize_riv,
			bounding_box) 


		self.riv_stages = []
		for row in range(nrow):
			for col in range(ncol):
				val = raster['array'][row][col]
				if val > 0 and ibound[0,row,col] > 0:
					self.riv_stages.append([1, row+1, col+1, 0, val, 0 ])

		# self.riv_stages = [
		# 	[1, 5, 5, 0, 1000, 0 ],
		# 	[1, 5, 7, 0, 1000, 0 ]
		# ]
		if plot:
			with open(data_dir+"riv.json", 'w') as f:
				f.write(json.dumps(raster))

    
    def perturb(self, mf, top, hk, nper, stage_height):

    	riv_stress_periods = []
    	for per in range(nper):
			riv_period = []
			
			for stage in self.riv_stages:
				row = stage[1]
				col = stage[2]
				riv_period.append([
					1, row, col,
					top[row,col] + 7 + stage_height[per], # stage height
					hk[row,col] * stage[4], # conductance
					top[row,col] + 5, # river bottom
					])

			riv_stress_periods.append(riv_period)

        return flopy.modflow.ModflowRiv(mf, 
        	layer_row_column_data=riv_stress_periods)
     

"""
wel 
"""     
class wel():
	def __init__(self, nlay, nrow, ncol, bounding_box):

		
		wells = [
		    # [ 150.61117315279262, -30.915837309134634], 
		    # [ 150.74639511003622, -30.93865809953413], 
		    [ 150.74673843279015, -30.885048778979645], 
		    [ 150.87479782523587, -30.876906574734466],
		    [ 150.91599655570462, -30.987047586140907],
		    # [ 150.8969893460744, 30.99850840015749],
		    # [ 150.69030904822284, -30.988504480642938],
		    [ 150.69695663871244, -31.005293991028097]
		]

		x_min = bounding_box[0][1]
		x_max = bounding_box[1][1]
		y_min = bounding_box[1][0]
		y_max = bounding_box[0][0]

		pixelWidth = (x_max - x_min)/ncol
		pixelHeight = (y_max - y_min)/nrow

		self.rasterized_wells = []
		for well in wells:
		    self.rasterized_wells.append([1, int((well[1]-y_min)/pixelHeight)+1, int((well[0]-x_min)/pixelWidth)+1 ])


		# unneccessary (just to visualize)
		if plot: 
			with open(data_dir+'wells.json', 'w') as f:
			    f.write(json.dumps( { "type": "FeatureCollection",
			    "features": [
			      { "type": "Feature",
			        "geometry": {"type": "Point", "coordinates": p},
			        "properties": {"flux": 100},
			        } for p in wells ]
			     } ))

			def rasterize_well(source_layer, clipping_poly):
			    properties = np.array([
			        feature.GetField("flux")
			        for feature in source_layer
			        if feature.GetGeometryRef().Clone().Within(clipping_poly)
			    ])
			    return np.sum(properties)

			with open(data_dir+"rasterize_well.json", 'w') as f:
			    f.write(json.dumps( ogr_to_modflow.ogr_to_modflow(
			    data_dir+"wells.json",
			    nrow, ncol,
			    rasterize_well,
			    bounding_box) 
			    ))
		#

	def perturb(self, mf, nper, rates):
		
		wel_stress_periods = []
		for per in range(nper):
			wells = []
			for well in self.rasterized_wells:
				wells.append([
					1, well[1], well[2], rates[per]
					])
			wel_stress_periods.append(wells)

		return flopy.modflow.ModflowWel(mf, layer_row_column_Q=wel_stress_periods)

# TODO
class rch():
    def __init__(self, nlay, nrow, ncol, bounding_box):


		# nrow, ncol, month
		def rasterize_rain(source_layer, clipping_poly):
			centroid = clipping_poly.Centroid()

			properties = [
				(feature.GetGeometryRef().Distance(centroid), feature.GetField('rainfall'))
				for feature in source_layer
			]
			sorted_properties = np.sort(
				np.array(
					properties, 
					dtype = [('distance', float), ('rain', list)]
				), 
				order='distance'
			)
			return sorted_properties[0]['rain']


		raster_rain = ogr_to_modflow.ogr_to_modflow(
			data_dir+"rainfall.json",
			nrow, ncol,
			rasterize_rain,
			bounding_box
		) 

		self.rain = np.rollaxis( np.array(raster_rain["array"]), 2 ) # roll to month, nrow,ncol

    
    def perturb(self, mf, intensity):

    	# TODO decide multiply factor, change sp to months
    	# 30 days per month, 1000mm in a m, 0.1% rainfall is effective
    	rech = intensity*self.rain[:3]/(30 * 1000 * 1000) 
        return flopy.modflow.ModflowRch(mf, rech = rech.tolist())  