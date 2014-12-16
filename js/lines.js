
function line(input,el){

  var lines = input.data;

  // Set the dimensions of the canvas / graph
  var margin = {top: 20, right: 10, bottom: 30, left: 45},
      width = el.node().clientWidth - margin.left - margin.right,
      height = 0.75*el.node().clientWidth - margin.top - margin.bottom;

  // Parse the date / time
  // var parseDate = d3.time.format("%d-%b-%y").parse;
  //   data.forEach(function(d) {
  //       d.date = parseDate(d.date);
  //       d.close = +d.close;
  //   });


  // Set the ranges
  // var x = d3.time.scale().range([0, width]);
  var x = d3.scale.log().range([0, width]);
  // var y = d3.scale.linear().range([height, 0]);
  var y = d3.scale.log().range([height, 0]);


  // Define the axes
  var xAxis = d3.svg.axis().scale(x)
      .orient("bottom")
      .ticks(4);

  var yAxis = d3.svg.axis().scale(y)
      .orient("left")
      .ticks(4);

  // Define the line
  var valueline = d3.svg.line()
      .x(function(d) { return x(d.x); })
      .y(function(d) { return y(d.y); });
      
  // Adds the svg canvas
  var svg = el
      .append("svg").attr('class', 'line_chart')
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", 
          "translate(" + margin.left + "," + margin.top + ")");


  x.domain([
    d3.min(lines, function(l){
      return d3.min(l.data, function(d) { return d.x; });
    }), 
    d3.max(lines, function(l){
      return d3.max(l.data, function(d) { return d.x; });
    })
  ]);

  y.domain([
    d3.min(lines, function(l){
      return d3.min(l.data, function(d) { return d.y; });
    }), 
    d3.max(lines, function(l){
      return d3.max(l.data, function(d) { return d.y; });
    })
  ]);


  // Scale the range of the data
  // x.domain(d3.extent(data, function(d) { return d.x; }));
  // y.domain(d3.extent(data, function(d) { return d.y; }));
  // y.domain([0.00000001, d3.max(data, function(d) { return d.y; })]);




  for (var i = 0; i < lines.length; i++) {

      for (var j = 0; j < lines[i].data.length; j++) {
        
        svg.append("circle")
          .datum(lines[i].data[j])
          .attr("cx", function(d) { return x(d.x); })
          .attr("cy", function(d) { return y(d.y); })
          .style('fill', color(lines[i].name))
          .attr("r", 2);

      };

          // .style("fill", function(d) { return color(d.z); });

      // Add the valueline path.
      svg.append("path")
          .attr("class", "line")
          .attr("d", valueline(lines[i].data))
          .style('stroke', color(lines[i].name));

      svg.append("text")
          .attr("transform", "translate(" + (width+3) + "," + y(lines[i].data.slice(-1)[0].y) + ")")
          .attr("dy", ".35em")
          .attr("text-anchor", "end")
          // .style("fill", "red")
          .text(lines[i].name);
  };


      // Add the X Axis
      svg.append("g")
          .attr("class", "x axis")
          .attr("transform", "translate(0," + height + ")")
          .call(xAxis)
               // .selectAll(".tick text")
            // .text(null)
          // .filter(powerOfTen)
          //   .text(10)
          // .append("tspan")
          //   .attr("dy", "-.7em")
          //   .text(function(d) { return Math.round(Math.log(d) / Math.LN10); });

      // Add the Y Axis
      svg.append("g")
          .attr("class", "y axis")
          .call(yAxis)
          // .selectAll(".tick text")
            // .text(null)
          // .filter(powerOfTen)
          //   .text(10)
          // .append("tspan")
          //   .attr("dy", "-.7em")
          //   .text(function(d) { return Math.round(Math.log(d) / Math.LN10); });

        function powerOfTen(d) {
          return d / Math.pow(10, Math.ceil(Math.log(d) / Math.LN10 - 1e-12)) === 1;
        }


  svg.append('text').text(input.labels.x).style("text-anchor", "middle").attr("transform", "translate(" + (width/2) + "," + (height+25) + ")");
  svg.append('text').text(input.labels.y).style("text-anchor", "middle").attr("transform", "translate(" + -35 + "," + (height/2) + ")rotate(-90)");
  svg.append('text').text(input.labels.title).style("text-anchor", "middle").style('font','14px sans-serif').attr("transform", "translate(" + (width/2) + "," + -5 + ")");





// embed styles so we can download svg 
  svg.selectAll('path').style({"fill": "none", "stroke-width": 2 });
  svg.selectAll('.axis line').style({"fill": "none", "stroke": "grey", "stroke-width": 1, "shape-rendering": "crispEdges" });
  svg.selectAll('.axis path').style({"fill": "none", "stroke": "grey", "stroke-width": 1, "shape-rendering": "crispEdges" });
  svg.style('font','10px sans-serif');

  // svg.selectAll('.axis path').style('display', 'none');
  // svg.selectAll('.frame').style({'stroke': '#aaa', 'fill': 'none'});
  // svg.selectAll('.circle').style('fill-opacity',.7);
  // svg.selectAll('.extent').style({ 'fill': '#000', 'fill-opacity': .125, 'stroke': '#fff' });


// .line_chart path { 
//     stroke: steelblue;
//     stroke-width: 2;
//     fill: none;
// }

// .line_chart .axis path,
// .line_chart .axis line {
//     fill: none;
//     stroke: grey;
//     stroke-width: 1;
//     shape-rendering: crispEdges;
// }








}
