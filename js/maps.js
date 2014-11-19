

var YlOrRd = ["#ffffcc","#ffeda0","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#bd0026","#800026"];
var Blues = ["#f7fbff","#deebf7","#c6dbef","#9ecae1","#6baed6","#4292c6","#2171b5","#08519c","#08306b"];
var Greens = ["#f7fcf5","#e5f5e0","#c7e9c0","#a1d99b","#74c476","#41ab5d","#238b45","#006d2c","#00441b"];
var Reds = ["#fff5f0","#fee0d2","#fcbba1","#fc9272","#fb6a4a","#ef3b2c","#cb181d","#a50f15","#67000d"];

// var BlGrn = Blues.reverse().slice(0,-1).concat(Greens);


// d3.select('body')
//   .selectAll('div.basdfasd')
//   .data(BlGrn)
//   .enter().append('div')
//   .style('background-color', function(d){ return d; })
//   .style('width', '100px')
//   .style('height', '100px')

/* -----------------------------------------------
  feature map
----------------------------------------------- */

// create a map in the "map" div, set the view to a given place and zoom
var map = L.map('feature_map').setView([-31,150.8], 11);
// var map = L.map('feature_map').setView([39.74739, -105], 13);
var control = L.control.layers().addTo(map);

L.control.scale().addTo(map);

// add an OpenStreetMap tile layer
// L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
//     attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
// }).addTo(map);

map.on('click', function(e) {
    console.log("Lat, Lon : " + e.latlng.lat + ", " + e.latlng.lng)
});



var Stamen_Toner = L.tileLayer('http://{s}.tile.stamen.com/toner/{z}/{x}/{y}.png', {
  attribution: 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> | Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>',
  subdomains: 'abcd',
  minZoom: 0,
  maxZoom: 20
}).addTo(map);


// https://github.com/Leaflet/Leaflet.draw

function onEachFeature(feature, layer) {
  var popupContent = '<div class="popup">';
  if (feature.properties) {
    for (var k in feature.properties) {
        var v = feature.properties[k];
        popupContent += k + ': ' + v + '<br />';
    }
  }
  popupContent += '</div>';

  layer.bindPopup(popupContent);
}







d3.json('data/APPT250K_Contours_line.json', function(data){

  // junk = data;

  var contours = data.features.map(function(d){ return d.properties.ELEVATION; }).sort(function(a,b){ return a>b;});

  var contour_color = d3.scale.quantize()
  // .domain([ contours[Math.round(contours.length/10)], contours[Math.round(9*contours.length/10)] ])
  .domain(d3.extent(contours))
  .range(YlOrRd);

  control.addOverlay(
    L.geoJson(data, {
      onEachFeature: onEachFeature,
      style : function(feature){
        return {
          "color": contour_color(feature.properties.ELEVATION),
          "weight": 1,
          "opacity": 1
        }
      }
    }).addTo(map),
    "Contours"
  )

});


var river_color = d3.scale.ordinal()
  .domain(["Minor", "Major"])
  .range(Blues.slice(-2));

var river_width = d3.scale.ordinal()
  .domain(["Minor", "Major"])
  .range([1,5]);

d3.json('data/APPT250K_WatercourseLines_line.json', function(data){

  control.addOverlay(
    L.geoJson(data, {
      onEachFeature: onEachFeature,
      style : function(feature){
        return {
          "color": river_color(feature.properties.HIERARCHY),
          "weight": river_width(feature.properties.HIERARCHY),
          "opacity": 1
        }
      }
    }).addTo(map),
    "Water courses"
  )

});

d3.json('data/wells.json', function(data){

  control.addOverlay(
    L.geoJson(data, {
      onEachFeature: onEachFeature,

      pointToLayer: function (feature, latlng) {
        return L.circleMarker(latlng, {
          radius: 6,
          fillColor: "#1a1a1a",
          color: "#000",
          weight: 1,
          opacity: 1,
          fillOpacity: 0.8
        });
      }
    }).addTo(map), 
  "NSW Water sites");

});



/* -----------------------------------------------
  generic loading of geojson -- good default/start
----------------------------------------------- */

// could use https://github.com/calvinmetcalf/leaflet.shapefile
//    or https://github.com/calvinmetcalf/leaflet.filegdb

var files = [
              // '  ', 
              // 'APPT250K_Contours_line.json',
              // "AHGFSurficialHydrogeologicUnit.json",
              "IGWWaterTableYield.json",
              "AHGFWaterTableAquifer.json",
              // "IGWAquiferYield.json",
              'GeologicUnitPolygons1M.json',
              "IGWWaterTableHydraulicConductivity.json",
              "rainfall.json",
              "nsw_sites.json",
              "NGIS_Bores.json",
              "perturbed_boundary.json",
              "aquifer_boundary.json",
              "simplified_aquifer_boundary.json",
              'clipped_GeologicUnitPolygons1M.json'
              ];

function load_geojson(file){
  d3.json('data/'+file, function(data){
    control.addOverlay(
      L.geoJson(data, {
        onEachFeature: onEachFeature,
      }),
      file
    )
  });
}

for (var i = files.length - 1; i >= 0; i--) {
  load_geojson(files[i]);
};


/* -----------------------------------------------
  modflow map
----------------------------------------------- */

// create a map in the "map" div, set the view to a given place and zoom
var modflow_map = L.map('modflow_map').setView([-31,150.8], 11);
var modflow_control = L.control.layers().addTo(modflow_map);

L.control.scale().addTo(modflow_map);

// add an OpenStreetMap tile layer
L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
}).addTo(modflow_map);


map.sync(modflow_map);
modflow_map.sync(map);


// TODO legend for d3_layer


var contour_color = d3.scale.log()
.range(d3.extent(YlOrRd));

// d3.json("data/ibound.json", function(data){ new d3_layer(data, "d3 ibound", d3.scale.linear().range(["white", "black"])); });
d3.json("data/perturbed_ibound.json", function(data){ new d3_layer(data, "d3 perturbed ibound", modflow_control, d3.scale.linear().range(["white", "black"])); });
d3.json("data/unperturbed_ibound.json", function(data){ new d3_layer(data, "d3 unperturbed ibound", modflow_control, d3.scale.linear().range(["white", "black"])); });
d3.json("data/hk_mean.json", function(data){ new d3_layer(data, "d3 hk", modflow_control, d3.scale.log().range(d3.extent(YlOrRd))); });
d3.json("data/sy_mean.json", function(data){ new d3_layer(data, "d3 sy", modflow_control, d3.scale.linear().range(d3.extent(YlOrRd))); });
d3.json("data/top.json", function(data){ new d3_layer(data, "d3 top", modflow_control, d3.scale.quantize().range(YlOrRd)); });
d3.json("data/rasterize_well.json", function(data){ new d3_layer(data, "d3 well", modflow_control, d3.scale.quantize().range(YlOrRd)); });
d3.json("data/riv.json", function(data){ new d3_layer(data, "d3 river", modflow_control, d3.scale.quantize().range(Blues)); });





// d3.json('data/grid.json', function(data){
//   modflow_control.addOverlay(
//     L.geoJson(data, {
//       onEachFeature: onEachFeature,
//     }).addTo(modflow_map),
//     'grid'
//   )
// });
