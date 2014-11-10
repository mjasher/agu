// https://github.com/Leaflet/Leaflet/blob/master/src/layer/GeoJSON.js
// thanks https://gist.github.com/ZJONSSON/3087431
// better than http://leafletjs.com/reference.html#ilayer
// Add a fake GeoJSON line to coerce Leaflet into creating the <svg> tag that d3_geoJson needs

var fake = L.geoJson({"type": "LineString","coordinates":[[0,0],[0,0]]}).addTo(modflow_map);
modflow_control.addOverlay(fake, "D3 svg");
// var g = fake._layers[Object.keys(fake._layers)[0]]._container;
// var svg = d3.select("#feature_map").select("svg");

var d3_layer = {

  initialize: function(){
    // this._el = svg
    //     .append('g')
    //     .append('rect');

    this._el = d3.select(fake._layers[Object.keys(fake._layers)[0]]._container)
        .append('rect');

    this._reset();

    modflow_map.on('viewreset', this._reset, this);
  },

  _latlng : new L.LatLng(-31.02,151),

  _bottomRight: { lat: -31.02+0.01, lng: 151+0.02},
  
  _dims: { width: 0.01, height: 0.02}, // Math.pow(2,modflow_map.getZoom())

  _reset: function(){
    var topLeft = modflow_map.latLngToLayerPoint(this._latlng);
    var bottomRight = modflow_map.latLngToLayerPoint(this._bottomRight);


    this._el
      .attr('x', topLeft.x)
      .attr('y', topLeft.y)
      .attr('width', bottomRight.x - topLeft.x)
      .attr('height', topLeft.y - bottomRight.y);
  }

}

d3_layer.initialize();



var d3_layer = {

  initialize: function(geojson, name, color_func){

    junk = this;
    this._color_func = color_func;

    var fake = L.geoJson({"type": "LineString","coordinates":[[0,0],[0,0]]}).addTo(modflow_map);
    modflow_control.addOverlay(fake, name);

    this._el = d3.select(fake._layers[Object.keys(fake._layers)[0]]._container);
    
    var this_guy = this;

    d3.json(geojson, function(data){

      var max = d3.max(data.array.map(function(d){ return d3.max(d); }));
      var min = d3.min(data.array.map(function(d){ return d3.min(d); }));
      this_guy._color_func.domain([min,max]);

      this_guy._data = [];

      // junky = this_guy._data;

      // make data easy for d3
      var nrow = data.array.length;
      var ncol = data.array[0].length;
      console.log("nrow,ncol",nrow,ncol, data.bottomLeft);
      for (var i = 0; i < nrow; i++) {
        for (var j = 0; j < ncol; j++) {
          this_guy._data.push({ 
            topLeft: {lat: data.bottomLeft.lat+(i+1)*data.pixelHeight, lng: data.bottomLeft.lng+(j)*data.pixelWidth},
            bottomRight: {lat: data.bottomLeft.lat+(i)*data.pixelHeight, lng: data.bottomLeft.lng+(j+1)*data.pixelWidth},
            value: data.array[i][j] 
          });
          this_guy._el.append('rect');
        };
      };
      
      this_guy._reset();


    });

    modflow_map.on('viewreset', this._reset, this);
  },

  _reset: function(){

    var this_guy = this;

    this._data.forEach(function(d){
        d.pix_topLeft = modflow_map.latLngToLayerPoint(d.topLeft);
        d.pix_bottomRight = modflow_map.latLngToLayerPoint(d.bottomRight);
    });

    this._el.selectAll('rect')
      .data(this._data)
      .attr('x', function(d) { return d.pix_topLeft.x; })
      .attr('y', function(d) { return d.pix_topLeft.y; })
      .attr('width', function(d){ return d.pix_bottomRight.x - d.pix_topLeft.x; })
      .attr('height', function(d){ return d.pix_bottomRight.y - d.pix_topLeft.y; })
      .attr('fill', function(d){ 
        if (d.value == 0) return 'rgba(0,0,0,0)';
        // else return this_guy._color_func(d.value); 
        return this_guy._color_func(d.value); 
      });

  }

}