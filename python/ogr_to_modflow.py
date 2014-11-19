from osgeo import ogr, osr, gdal
import numpy as np
import json
import os
import subprocess

"""
routines to convert data in ogr format (shp, geojson, kml, gdb ....) to MODFLOW input arrays
michael.james.asher@gmail.com
November 2014
==================================

crucial reference
gdal.org/python/osgeo.ogr.Geometry-class.html
"""

# converts source layer to modflow array using given a custom function
#  rasterize(layer, clipping poly) which takes the layer and polygon for once cell and returns cell's value
def ogr_to_modflow( source_file, nrow, ncol, rasterize, bounding_box ):
	# open source (file)
	source_ds = ogr.Open(source_file)
	source_layer = source_ds.GetLayer()

	# to ensure cut geometries (like ibound) remain consistent
	# x_min, x_max, y_min, y_max = source_layer.GetExtent()
	# print x_min == bounding_box[0][1], x_max == bounding_box[1][1], y_min == bounding_box[1][0], y_max == bounding_box[0][0]
	x_min = bounding_box[0][1]
	x_max = bounding_box[1][1]
	y_min = bounding_box[1][0]
	y_max = bounding_box[0][0]

	pixelWidth = (x_max - x_min)/ncol
	pixelHeight = (y_max - y_min)/nrow

	array = np.empty((nrow,ncol)).tolist()
	for row in range(nrow):
		for col in range(ncol):

			# Create clipping polygon
			ring = ogr.Geometry(ogr.wkbLinearRing)
			ring.AddPoint(x_min+(col)*pixelWidth, y_min+(row)*pixelHeight)
			ring.AddPoint(x_min+(col+1)*pixelWidth, y_min+(row)*pixelHeight)
			ring.AddPoint(x_min+(col+1)*pixelWidth, y_min+(row+1)*pixelHeight)
			ring.AddPoint(x_min+(col)*pixelWidth, y_min+(row+1)*pixelHeight)
			ring.AddPoint(x_min+(col)*pixelWidth, y_min+(row)*pixelHeight)
			poly = ogr.Geometry(ogr.wkbPolygon)
			poly.AddGeometry(ring)
			
			# speeds things up but breaks them too
			# source_layer.SetSpatialFilterRect(x_min+col*pixelWidth, x_min+(col+1)*pixelWidth, y_min+row*pixelHeight, y_min+(row+1)*pixelHeight)

			source_layer.ResetReading()

			array[row][col] = rasterize(source_layer, poly)

	return {
		"bottomLeft": { "lat": y_min, "lng": x_min },
		"pixelHeight": pixelHeight,
		"pixelWidth": pixelWidth,
		"array": array 
	}



def projected_distance(from_epsg, to_epsg, p1, p2):

	inSpatialRef = osr.SpatialReference()
	inSpatialRef.ImportFromEPSG(from_epsg)

	outSpatialRef = osr.SpatialReference()
	outSpatialRef.ImportFromEPSG(to_epsg)

	# create the CoordinateTransformation
	transform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)	

	p1.Transform(transform)
	p2.Transform(transform)
	return p1.Distance(p2)




# builds a geojson grid
def build_grid(source_file, dest_file, nrow, ncol):

	if os.path.exists(dest_file):
		os.remove(dest_file)

	# open source (file)
	source_ds = ogr.Open(source_file)
	source_layer = source_ds.GetLayer()

	x_min, x_max, y_min, y_max = source_layer.GetExtent()

	pixelWidth = (x_max - x_min)/ncol
	pixelHeight = (y_max - y_min)/nrow

	# Create the output Driver
	outDriver = ogr.GetDriverByName('GeoJSON')

	# Create the output GeoJSON
	outDataSource = outDriver.CreateDataSource(dest_file)
	outLayer = outDataSource.CreateLayer('grid' )

	# Get the output Layer's Feature Definition
	featureDefn = outLayer.GetLayerDefn()

	for row in range(nrow):
		for col in range(ncol):

			# Create clipping polygon
			ring = ogr.Geometry(ogr.wkbLinearRing)
			ring.AddPoint(x_min+(col)*pixelWidth-0.000, y_min+(row)*pixelHeight-0.000)
			ring.AddPoint(x_min+(col+1)*pixelWidth+0.000, y_min+(row)*pixelHeight-0.000)
			ring.AddPoint(x_min+(col+1)*pixelWidth+0.000, y_min+(row+1)*pixelHeight+0.000)
			ring.AddPoint(x_min+(col)*pixelWidth-0.000, y_min+(row+1)*pixelHeight+0.000)
			ring.AddPoint(x_min+(col)*pixelWidth-0.000, y_min+(row)*pixelHeight-0.000)
			poly = ogr.Geometry(ogr.wkbPolygon)
			poly.AddGeometry(ring)

			# create a new feature
			outFeature = ogr.Feature(featureDefn)

			# Set new geometry
			outFeature.SetGeometry(poly)

			# Add new feature to output Layer
			outLayer.CreateFeature(outFeature)

			# destroy the feature
			outFeature.Destroy()

	# Close DataSources
	outDataSource.Destroy()

# might have to segmentize

# Approach 1: create memory layer, add rectangle, use layer.clip(mem_layer)
# Approach 2: use geom.intersection(geom)
# Note source_layer.SetSpatialFilterRect doesn't clip

def further_clip(source_file, dest_file, nrow, ncol, bounding_box):

	x_min = bounding_box[0][1]
	x_max = bounding_box[1][1]
	y_min = bounding_box[1][0]
	y_max = bounding_box[0][0]

	pixelWidth = (x_max - x_min)/ncol
	pixelHeight = (y_max - y_min)/nrow

	if os.path.exists(dest_file):
		os.remove(dest_file)

	subprocess.call([
		"ogr2ogr",
		"-clipsrc", str(bounding_box[0][1]+pixelWidth), str(bounding_box[0][0]-pixelHeight), 
		str(bounding_box[1][1]-pixelWidth), str(bounding_box[1][0]+pixelHeight),
		"-f", "GeoJSON", dest_file, source_file,
		"-t_srs", "EPSG:4326" 
		])

def create_ibound():
	# create ibound geometry
	source_dir = '/home/mikey/Dropbox/pce/interactions/poster/data/'
	filename = source_dir + 'aquifer_boundary.json'

	# combine all geological regions from further clipped file (with buffer for perturbation) exclude Namoi formation
	# geology_ds = ogr.Open(source_dir+'GeologicUnitPolygons1M.json')
	geology_ds = ogr.Open(source_dir+'clipped_GeologicUnitPolygons1M.json')
	geology_layer = geology_ds.GetLayer()

	geos = [feature.GetGeometryRef().Clone() for feature in geology_layer if feature.GetGeometryRef() and feature.GetField('NAME') not in ["Namoi Formation"]]
	union = geos[0]
	for geo in geos[1:]:
		union = union.Union(geo)

	# exclude regions where SpecYield is undefined
	hk_ds = ogr.Open(source_dir+"IGWWaterTableHydraulicConductivity.json")
	hk_layer = hk_ds.GetLayer()	

	geos = [feature.GetGeometryRef().Clone() for feature in hk_layer if feature.GetField('SpecYield') == None]
	for geo in geos:
		union = union.Difference(geo)

	# just use the biggest polygon in the multipolygon (polygon collection)
	max_area = 0
	the_poly =  None
	for poly in union:
		if poly.Area() > max_area:
			max_area = poly.Area()
			the_poly = poly

	union = union.Intersection(the_poly)

	if os.path.exists(filename):
	    os.remove(filename)

	# write union to file
	outDriver = ogr.GetDriverByName('GeoJSON')
	outDataSource = outDriver.CreateDataSource(filename)
	outLayer = outDataSource.CreateLayer('ibound' )

	featureDefn = outLayer.GetLayerDefn()

	outFeature = ogr.Feature(featureDefn)
	outFeature.SetGeometry(union)
	outLayer.CreateFeature(outFeature)

	outFeature.Destroy()
	outDataSource.Destroy()



if __name__ == '__main__':

	nrow, ncol = (20,30)
	source_dir = '/home/mikey/Dropbox/pce/interactions/poster/data/'
	bounding_box = [[-30.82,150.54],[-31.04,151.01]] #  top left, bottom right

	# ibound
	# ---------------------

	further_clip(source_dir+'GeologicUnitPolygons1M.json',
						 source_dir+'clipped_GeologicUnitPolygons1M.json',
						 nrow, ncol, bounding_box)
	create_ibound() # writes ibound.json
	
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


	def rasterize_well(source_layer, clipping_poly):
		properties = np.array([
			feature.GetField("flux")
			for feature in source_layer
			if feature.GetGeometryRef().Clone().Within(clipping_poly)
		])
		return np.sum(properties)

	with open(source_dir+"rasterize_well.json", 'w') as f:
			f.write(json.dumps( ogr_to_modflow(
				source_dir+"wells.json",
				nrow, ncol,
				rasterize_well,
				bounding_box) 
			))



