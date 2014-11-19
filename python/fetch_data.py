import subprocess
import os
import csv
import json
import re

'''
routines to fetch, reproject, convert and clip data ready for GW modelling
michael.james.asher@gmail.com
October 2014
==================================
'''

# this should really be a bash script, but .py looks friendly
# note many of GA's WFS could be easily "scraped" using http://anitagraser.com/2012/09/29/wfs-to-postgis-in-one-step/

destination_dir = '/home/mikey/Dropbox/pce/agu/data/'

# should include a buffer
bounding_box = [[-30.82,150.54],[-31.04,151.01]] #  top left, bottom right

# get surface geology shapefiles from GA
# ------------------------------------------------------
# GeologicUnitPolygons1M from http://www.ga.gov.au/metadata-gateway/metadata/record/gcat_c8856c41-0d5b-2b1d-e044-00144fdd4fa6/Surface+Geology+of+Australia+1%3A1+million+scale+dataset+2012+edition
# same as "AHGFSurficialHydrogeologicUnit.json"
def surface():
	for file_name in ["/home/mikey/ABS/GA/surface geology/GeologicUnitPolygons1M.shp"]:
		subprocess.call([
			"ogr2ogr",
			"-clipsrc", str(bounding_box[0][1]), str(bounding_box[0][0]), str(bounding_box[1][1]), str(bounding_box[1][0]),
			"-f", "GeoJSON", destination_dir+os.path.basename(file_name).replace(".shp",".json"), file_name,
			"-t_srs", "EPSG:4326" 
			])

# get river shapefile for region from mapconnect
# ------------------------------------------------------
# watercourse lines from hydrography from http://mapconnect.ga.gov.au/MapConnect/?site=250K&accept_agreement=on
# 3857 vs 900913 ?

def mapconnect():
	for file_name in ["/home/mikey/ABS/mapconnnect/APPT250K_WatercourseLines_line.shp", 
						"/home/mikey/ABS/mapconnnect/APPT250K_Contours_line.shp"]:
		subprocess.call([
			"ogr2ogr",
			"-clipsrc", str(bounding_box[0][1]), str(bounding_box[0][0]), str(bounding_box[1][1]), str(bounding_box[1][0]),
			"-f", "GeoJSON", destination_dir+os.path.basename(file_name).replace(".shp",".json"), file_name,
			"-t_srs", "EPSG:4326" 
			])

# get river gauges and bore logs from NSW realtime water data
# ------------------------------------------------------
# fetch_nsw_water_data.py module automates http://realtimedata.water.nsw.gov.au/water.stm

def nsw():
	import fetch_nsw_water_data
	
	# get all the sites in a bounding box and write them to a geojson
	sites = fetch_nsw_water_data.fetch_sites(bounding_box) 
	
	fetch_nsw_water_data.write_sites(sites, destination_dir+'nsw_sites.json')

	# for each site, get details of all available time series
	# variable_by_site = fetch_nsw_water_data.site_variables(sites)

	# for each site, get all timeseries (will take > 10 min)
	# by_site = get_all_ts(variable_by_site)



# get geofabric 
# ------------------------------------------------------
# http://geofabric.bom.gov.au/documentation/
# download from http://data.gov.au/dataset/australian-hydrological-geospatial-fabric-geofabric-groundwater-cartography-v2-x
# nice interface at http://nationalmap.nicta.com.au/


# get layers 
#	ogrinfo /home/mikey/ABS/BOM/GW_Cartography_GDB/GW_Cartography.gdb 
def geofabric():
	file_name = "/home/mikey/ABS/BOM/GW_Cartography_GDB/GW_Cartography.gdb"
	for layer in ["AHGFSurficialHydrogeologicUnit", "IGWWaterTableHydraulicConductivity", "IGWWaterTableYield", "AHGFWaterTableAquifer", "IGWAquiferYield" ]:
		subprocess.call([
			"ogr2ogr",
			"-clipsrc", str(bounding_box[0][1]), str(bounding_box[0][0]), str(bounding_box[1][1]), str(bounding_box[1][0]),
			"-f", "GeoJSON", destination_dir + layer+".json", file_name, layer,
			"-t_srs", "EPSG:4326" 
			])


# get rainfall off BOM
# ------------------------------------------------------
# find nearest sites http://www.bom.gov.au/climate/data/index.shtml?bookmark=136
# could make scraper using ftp://ftp.bom.gov.au/anon2/home/ncc/metadata/sitelists/stations.zip and eg. http://www.bom.gov.au/clim_data/cdio/tables/text/IDCJCM0037_066196.csv
# http://www.bom.gov.au/climate/ also has grids
def bom():
	import bom_to_geojson
	directory = "/home/mikey/ABS/BOM/climate/"
	files = [directory+f for f in os.listdir(directory) if f.endswith('.csv')]
	bom_to_geojson.bom_to_geojson(files, destination_dir+'rainfall.json')
	

# get lithology logs from NGIS
# download ngis_shp_nsw from ftp://ftp.bom.gov.au/anon/home/water/ngis/downloads/shp/
def ngis():

	# clip bore shapefile
	
	source = "/home/mikey/ABS/BOM/ngis_shp_NSW/ngis_shp_NSW/NGIS_Bores.shp"
	dest = destination_dir + "NGIS_Bores.json"

	try:
		os.remove(dest)
	except:
		pass

	# note clip dst instead
	subprocess.call([
			"ogr2ogr",
			"-clipdst", str(bounding_box[0][1]), str(bounding_box[0][0]), str(bounding_box[1][1]), str(bounding_box[1][0]),
			"-f", "GeoJSON", dest, source,
			"-t_srs", "EPSG:4326"
			])

	# get a set of bore ids in clipped layer

	bore_ids = set()
	with open(dest) as json_file:
		geojson = json.load(json_file)
		for feature in geojson["features"]:
			bore_ids.add(feature["properties"]["HydroID"])


	# make csv containing lithology logs relating to bore ids in clipped layer

	source = "/home/mikey/ABS/BOM/ngis_shp_NSW/ngis_shp_NSW/NGIS_LithologyLog.dbf"
	dest1 = destination_dir + "NGIS_Lithology_unfiltered.csv"
	dest2 = destination_dir + "NGIS_Lithology.csv"

	try:
		os.remove(dest1)
	except:
		pass

	subprocess.call([
		"ogr2ogr",
		"-f", "CSV", dest1, source,
		])

	# print bore_ids
	with open(dest1) as read_f:
		head = csv.reader(read_f).next()

	by_ids = {}
	with open(dest1) as read_f, open(dest2,'w') as write_f:

		reader = csv.DictReader(read_f)
		writer = csv.DictWriter(write_f, head)
		writer.writeheader()
		for row in reader:
			if int(row["BoreID"]) in bore_ids:
				if not int(row["BoreID"]) in by_ids:
					by_ids[int(row["BoreID"])] = []
				by_ids[int(row["BoreID"])].append(row)
				writer.writerow(row)


	# create 3d bores

	# print by_ids
	with open(dest) as json_file, open(destination_dir+"3d_bores.json",'w') as bores_3d:
		geojson = json.load(json_file)
		for feature in geojson["features"]:
			if feature["properties"]["HydroID"] in by_ids:
				lithology = by_ids[feature["properties"]["HydroID"]]

				feature["geometry"]["type"] = "MultiLineString"
				point = feature["geometry"]["coordinates"]
				feature["geometry"]["coordinates"] = [ [ [point[0],point[1],float(l["TopElev"])], 
													[point[0],point[1],float(l["BottomElev"])] ] 
													for l in lithology ]
			else:
				print "couldn't", feature["properties"]["HydroID"] 
				pass

		bores_3d.write(json.dumps(geojson))



# get namoi shp from GDE atlas
# ------------------------------------------------------
# ftp://ftp.bom.gov.au/anon/home/water/gde/river_basin/
# snoop around BOM
# ftp://ftp.bom.gov.au/anon/home/water/ngis/downloads/shp/
# ftp://ftp.bom.gov.au/anon/home/geofabric/


# get vegitation cover
# ------------------------------------------------------
# http://www.environment.gov.au/topics/science-and-research/databases-and-maps/national-vegetation-information-system/data-product-0



if __name__ == '__main__':
	# surface()
	# mapconnect()
	# nsw()
	# geofabric()
	# bom()
	ngis()