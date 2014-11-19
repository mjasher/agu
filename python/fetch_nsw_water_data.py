import urllib2
import json
import os

# routines to fetch data from from NSW water
# michael.james.asher@gmail.com
# October 2014
# ==================================

# see WIR user guide 
# http://webcache.googleusercontent.com/search?q=cache:zczMmAj4stUJ:kisters.com.au/doco/hydllp.htm+&cd=2&hl=en&ct=clnk&gl=au&client=ubuntu
# Dates take the form yyyymm22hhiiee


# get data from NSW water
# http://realtimedata.water.nsw.gov.au/cgi/webservice.server.pl?jsoncallback=printData&{"params":{"return_type":"array","sitelist_filter":"FILTER(TSFILES(PROV),MATCH(210*))","field_list":["station","stname","shortname","latitude","longitude","elev"],"table_name":"site"},"function":"get_db_info","version":2}
# https://github.com/tonycaine/pages-for-appchallenge/blob/master/sites-inSiteList.html


# fetch bores/gauges in bounding box
# ----------------------------------
def fetch_sites(bounding_box):
	url = "http://realtimedata.water.nsw.gov.au/cgi/webservice.server.pl?"
	#params = "jsoncallback=printData&{%22params%22:{%22return_type%22:%22array%22,%22sitelist_filter%22:%22FILTER%28TSFILES%28PROV%29,MATCH%28210*%29%29%22,%22field_list%22:[%22station%22,%22stname%22,%22shortname%22,%22latitude%22,%22longitude%22,%22elev%22],%22table_name%22:%22site%22},%22function%22:%22get_db_info%22,%22version%22:2}"

	parameters = {
		"params":{
			"return_type": "array",
			# "sitelist_filter": "FILTER(TSFILES(PROV),MATCH(210*))", 
			# "geo_filter":{"circle":["'+latitude+'","'+longitude+'","'+radiusdegree+'"]}, 
			"geo_filter":{"rectangle": bounding_box}, 
			# "field_list": ["station","stname","shortname","latitude","longitude","elev"],
			"table_name":"site"
		},
		"function": "get_db_info",
		"version": 2
	}

	params = "jsoncallback=printData&" + json.dumps(parameters)

	params = params.replace(' ','')
	print "fetch_sites params", params

	response = urllib2.urlopen(url+params).read()
	response = json.loads( response.replace('printData(','')[:-2] ) #");"

	return response

	
# write as geojson
# ----------------------------------
def write_sites(response, name):
	geojson = {"type": "FeatureCollection", "features": []}

	for row in response['_return']['rows']:
		geojson["features"].append({ 
				"type": "Feature",
				"geometry": {
					"type": "Point",
					"coordinates": [float(row["longitude"]), float(row["latitude"])]
				},
				"properties": row
			})

	with open(name, 'w') as bores:
		bores.write(json.dumps(geojson))


def get_datasources_by_site(site_list):
	url = 'http://realtimedata.water.nsw.gov.au/cgi/webservice.server.pl?jsoncallback=printData&'

	parameters = {
		"params":{
			"site_list": ','.join(site_list),
		},
		"function": "get_datasources_by_site",
		"version": 1
	}

	params = json.dumps(parameters).replace(' ','')
	print "get_datasources_by_site params", params

	response = urllib2.urlopen(url+params).read()
	print "get_datasources_by_site response", response
	return json.loads( response.replace('printData(','')[:-2] )


def get_variable_list(site_list, datasource):
	url = 'http://realtimedata.water.nsw.gov.au/cgi/webservice.server.pl?jsoncallback=printData&'

	parameters = {
		"params":{
			"site_list": ','.join(site_list),
			"datasource": datasource
		},
		"function": "get_variable_list",
		"version": 1
	}

	params = json.dumps(parameters).replace(' ','')
	print "get_variable_list params", params

	response = urllib2.urlopen(url+params).read()
	return json.loads( response.replace('printData(','')[:-2] )


# todo wrap everything in this
def generic_call(parameters):
	url = 'http://realtimedata.water.nsw.gov.au/cgi/webservice.server.pl?jsoncallback=printData&'
	params = json.dumps(parameters).replace(' ','')
	response = urllib2.urlopen(url+params).read()
	return  json.loads( response.replace('printData(','')[:-2] )


# for each site, get details of all available time series
#-------------------------------------
# TODO check this get's them all (buffer is not exceeded)
def site_variables(sites):
	variable_by_site = {}
	datasources_by_site = get_datasources_by_site([ s["station"] for s in sites["_return"]["rows"] ])
	for site in datasources_by_site["_return"]["sites"]:
		variable_by_site[site["site"]] = []
		for datasource in site["datasources"]:
			site_variables = get_variable_list([site["site"]], datasource)["_return"]["sites"][0]
			for variable in site_variables["variables"]:
				variable["datasource"] = datasource
				variable_by_site[site["site"]].append(variable)
				# print site_variables["site_details"]["name"], variable['name']
			assert(site["site"] == site_variables["site"])
			# print "site", site, "datasource", datasource, get_variable_list([site["site"]], datasource)
	return variable_by_site

# for each site, get all timeseries
#-------------------------------------
# takes a long time [Finished in 444.7s]
def get_all_ts(variable_by_site):
	by_site = {}
	for site in variable_by_site:
		tss = []
		for variable in variable_by_site[site]:
			ts =  generic_call({
				"params":{
					"site_list": site,
					"interval": "day",
					# "datasource": "A", #"CP"
					"datasource": variable['datasource'],
					"start_time": variable['period_start'],
					"end_time": variable['period_end'],
					"varfrom": variable['variable'],
					"varto": variable['variable'],
					"data_type":"mean",
					"multiplier":"1"
				},
				"function": "get_ts_traces",
				"version": 1
			})
			tss.append(ts)

		by_site[site] = tss

	return by_site


def fetch_available_ts(sites):

	site_list = ""
	for row in sites['_return']['rows']:
		site_list += row['station'] +','


	url = 'http://realtimedata.water.nsw.gov.au/cgi/webservice.server.pl?jsoncallback=printData&'
	#params = "jsoncallback=printData&{%22params%22:{%22return_type%22:%22array%22,%22sitelist_filter%22:%22FILTER%28TSFILES%28PROV%29,MATCH%28210*%29%29%22,%22field_list%22:[%22station%22,%22stname%22,%22shortname%22,%22latitude%22,%22longitude%22,%22elev%22],%22table_name%22:%22site%22},%22function%22:%22get_db_info%22,%22version%22:2}"

	parameters = {
		"params":{
			"site_list": site_list[:-1], #"055121,419075",
			"datasources": ["A","B","C", "T", "Z"], #"CP"
		},
		"function": "get_ts_blockinfo",
		"version": 1
	}

	params = json.dumps(parameters).replace(' ','')
	print "fetch_available_ts params", params

	response = urllib2.urlopen(url+params).read()

	try:
		return json.loads( response.replace('printData(','')[:-2] ) #");"
	except:
		print "fetch_available_ts response", response



# fetch bores/gauges in bounding box
# ----------------------------------
def fetch_timeseries():
	url = 'http://realtimedata.water.nsw.gov.au/cgi/webservice.server.pl?jsoncallback=printData&'
	#params = "jsoncallback=printData&{%22params%22:{%22return_type%22:%22array%22,%22sitelist_filter%22:%22FILTER%28TSFILES%28PROV%29,MATCH%28210*%29%29%22,%22field_list%22:[%22station%22,%22stname%22,%22shortname%22,%22latitude%22,%22longitude%22,%22elev%22],%22table_name%22:%22site%22},%22function%22:%22get_db_info%22,%22version%22:2}"

	parameters = {
		"params":{
			# "site_list": "FILTER(TSFILES(PROV),MATCH(21000*))", #["419075"],
			"site_list": "055121,419075",
			"interval": "day",
			"datasource": "A", #"CP"
			"start_time": 20000101000000,
			"end_time": 20150101000000,
			"varfrom": 100,
			"varto": 140,
			"data_type":"mean",
			"multiplier":"1"
		},
		"function": "get_ts_traces",
		"version": 1
	}

	params = json.dumps(parameters).replace(' ','')
	print "fetch_timeseries params", params

	response = urllib2.urlopen(url+params).read()
	return response

	# response = json.loads( response.replace('printData(','')[:-2] ) #");"

	# return response



# main
# ----------------------------------
if __name__ == '__main__':

	bounding_box = [[-30.87,150.54],[-31.04,151.01]] #  top left, bottom right

	# get all the sites in a bounding box and write them to a geojson
	#-------------------------------------
	# should use get_site_geojson
	sites = fetch_sites(bounding_box) 
	
	file_name = 'poster/data/nsw_sites.json'
	if not os.path.exists(file_name):
		write_sites(sites, file_name)


	# for each site, get details of all available time series
	#-------------------------------------
	variable_by_site = site_variables(sites)

	file_name = 'poster/data/variable_by_site.json'
	if not os.path.exists(file_name):
		with open(file_name,'w') as f:
			f.write( json.dumps(variable_by_site) )

	# for each site, get all timeseries
	#-------------------------------------
	# takes a long time [Finished in 444.7s]
	by_site = get_all_ts(variable_by_site)

	for site in by_site:
		file_name = 'poster/data/ts/'+site+'.json'
		if not os.path.exists(file_name):
			with open(file_name,'w') as f:
				f.write( json.dumps(by_site[site]) )

	# junk/alternatives
	# -----------------------------
	# print fetch_available_ts(sites)
	# print fetch_timeseries()
