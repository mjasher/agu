// d3 rendered leaflet layer
// another way: http://g0v.github.io/twgeojson/map.html 

// function d3_layer(geojson, name, color_func){
function d3_layer(data, name, control, color_func){
    
    this._reset = function(){


      var this_guy = this;

      this._data.forEach(function(d){
          d.pix_topLeft = control._map.latLngToLayerPoint(d.topLeft);
          d.pix_bottomRight = control._map.latLngToLayerPoint(d.bottomRight);
      });

      this._el.selectAll('rect')
        .data(this._data)
        .attr('x', function(d) { return d.pix_topLeft.x; })
        .attr('y', function(d) { return d.pix_topLeft.y; })
        .attr('width', function(d){ return d.pix_bottomRight.x - d.pix_topLeft.x; })
        .attr('height', function(d){ return d.pix_bottomRight.y - d.pix_topLeft.y; })
        .attr('fill', function(d){ 
          if (d.value == 0 || d.value == -9999) return 'rgba(0,0,0,0)';
          // else return this_guy._color_func(d.value); 
          return this_guy._color_func(d.value); 
        })
        .on('click', function(d){
          console.log(d.value);
        })

    }


    this._color_func = color_func;

    var fake = L.geoJson({"type": "LineString","coordinates":[[0,0],[0,0]]}).addTo(control._map);
    control.addOverlay(fake, name);

    this._el = d3.select(fake._layers[Object.keys(fake._layers)[0]]._container);
    
    var this_guy = this;

    // d3.json(geojson, function(data){

      // var max = d3.max(data.array.map(function(d){ return d3.max(d); }));
      var max = d3.max(data.array.map(function(d){ return d3.max(d.filter(function(e){ 
        return !(e == -9999 || e == 0); //! isNaN(e); 
      })); }));
      var min = d3.min(data.array.map(function(d){ return d3.min(d.filter(function(e){ 
        return !(e == -9999 || e == 0); 
      })); }));

      this_guy._color_func.domain([min,max]);

      this_guy._data = [];

      // junky = this_guy._data;

      // make data easy for d3
      var nrow = data.array.length;
      var ncol = data.array[0].length;
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


    // });

    control._map.on('viewreset', this._reset, this);

}