import csv
import json
import re

def bom_to_geojson(files, destination):
	rainfall_sites =  { "type": "FeatureCollection", "features": [] }
	for f_name in files:
		with open(f_name) as f:
			reader = csv.reader(f)
			for row in reader:
				if row:
					match = re.compile("Monthly Climate Statistics for (.*)").match(row[0])
					if match:
						name = match.group(1)

					match = re.compile("Latitude:   (\d+.\d+) Degrees South").match(row[0])
					if match:
						lat = match.group(1)

					match = re.compile("Longitude:  (\d+.\d+) Degrees East").match(row[0])
					if match:
						lng = match.group(1)

					match = re.compile("Mean rainfall \(mm\) for years.*").match(row[0])
					if match:
						mean_rainfall = row[1:14]
		rainfall_sites["features"].append({ 
				"type": "Feature",
		        "geometry": {"type": "Point", "coordinates": [float(lng), -float(lat)]},
		        "properties": { "rainfall": [float(r) for r in mean_rainfall], "name": name }
		        })

	with open(destination, 'w') as f:
		f.write(json.dumps(rainfall_sites))