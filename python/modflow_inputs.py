from osgeo import ogr, osr, gdal
import numpy as np
import json
import os
import subprocess

from ogr_to_modflow import ogr_to_modflow, further_clip, create_ibound

nrow, ncol = (20,30)
source_dir = '/home/mikey/Dropbox/pce/interactions/poster/data/'
bounding_box = [[-30.82,150.54],[-31.04,151.01]] #  top left, bottom right


# grid
# ---------------------

if not os.path.exists(source_dir+'grid.json'):
	build_grid(source_dir+"APPT250K_WatercourseLines_line.json", source_dir+'grid.json',
		nrow, ncol)


# hk, sy
# ---------------------
def rasterize_hk(source_layer, clipping_poly):
	properties = np.array([
		[feature.GetGeometryRef().Clone().Intersection(clipping_poly).Area(),
			feature.GetField('HydKValue')]
		for feature in source_layer
	])
	return np.average(properties[:,1], weights=properties[:,0])

with open(source_dir+"hk_mean.json", 'w') as f:
		f.write(json.dumps( ogr_to_modflow(
			source_dir+"IGWWaterTableHydraulicConductivity.json",
			nrow, ncol,
			rasterize_hk,
			bounding_box) 
		))

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

with open(source_dir+"sy_mean.json", 'w') as f:
		f.write(json.dumps( ogr_to_modflow(
			source_dir+"IGWWaterTableHydraulicConductivity.json",
			nrow, ncol,
			rasterize_sy,
			bounding_box) 
		))


# riv
# ---------------------

# transform so we can use length of river in meters
inDs = ogr.Open(source_dir+"APPT250K_WatercourseLines_line.json")
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
			# clipped.Transform(transform)
			properties.append(clipped.Length())
	return np.sum(properties)

with open(source_dir+"riv.json", 'w') as f:
		f.write(json.dumps( ogr_to_modflow(
			source_dir+"APPT250K_WatercourseLines_line.json",
			nrow, ncol,
			rasterize_riv,
			bounding_box) 
		))


# botm, top
# ---------------------

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


with open(source_dir+"top.json", 'w') as f:
		f.write(json.dumps( ogr_to_modflow(
			source_dir+"APPT250K_Contours_line.json",
			nrow, ncol,
			rasterize_top,
			bounding_box) 
		))

# ibound
# ---------------------

further_clip(source_dir+'GeologicUnitPolygons1M.json',
					 source_dir+'clipped_GeologicUnitPolygons1M.json',
					 nrow, ncol, bounding_box)
create_ibound() # writes ibound.json


def rasterize_ibound(source_layer, clipping_poly):

	properties = []
	for feature in source_layer:
		new_feature = feature.GetGeometryRef().Clone()

		properties.append(new_feature.Intersection(clipping_poly).Area())

	# properties = np.array([
	# 	feature.GetGeometryRef().Clone().Intersection(clipping_poly).Area()
	# 	for feature in source_layer
	# ])
	if np.sum(properties) == 0:
		return 0
	else:
		return 1

with open(source_dir+"ibound.json", 'w') as f:
		f.write(json.dumps( ogr_to_modflow(
			source_dir+'aquifer_boundary.json',
			nrow, ncol,
			rasterize_ibound,
			bounding_box)
		))
