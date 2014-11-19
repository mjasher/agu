

function scatter(el, scatter_data){

  var color = d3.scale.category10();

  // var width = document.getElementById('scatter').clientWidth,
  var width = el.node().clientWidth,
      padding = 50;

  var x = d3.scale.linear()
      .domain(d3.extent(scatter_data.map(function(d){ return d.x })))
      .range([padding / 2, width - padding / 2])
  var y = d3.scale.linear()
      .domain(d3.extent(scatter_data.map(function(d){ return d.y })))
      .range([width - padding / 2, padding / 2]);

  var xAxis = d3.svg.axis()
      .scale(x)
      .orient("bottom")
      .ticks(5)
      .tickSize(width-padding/2);

  var yAxis = d3.svg.axis()
      .scale(y)
      .orient("left")
      .ticks(5)
      .tickSize(width-padding/2);

  var svg = el.append("svg")
      .attr("width", width)
      .attr("height", width)
    .append("g")
      // .attr("transform", "translate(" + (padding/2) + "," + (padding/2) + ")");

  svg.append("g")
      .attr("class", "x axis")
      .call(xAxis);

  svg.append("g")
      .attr("class", "y axis")
      .attr("transform", "translate(" + width + ",0)")
      .call(yAxis);

  svg.append("rect")
        .attr("class", "frame")
        .attr("x", padding / 2)
        .attr("y", padding / 2)
        .attr("width", width - padding)
        .attr("height", width - padding);

  svg.selectAll("circle")
        .data(scatter_data)
      .enter().append("circle")
        .attr("cx", function(d) { return x(d.x); })
        .attr("cy", function(d) { return y(d.y); })
        .attr("r", 3)
        .style("fill", function(d) { return color(d.z); });


  svg.selectAll('.axis line').style('stroke', '#ddd');
  svg.selectAll('.axis path').style('display', 'none');
  svg.selectAll('.frame').style({'sroke': '#aaa', 'fill': 'none'});
  svg.selectAll('.circle').style('fill-opacity',.7);
  svg.selectAll('.extent').style({ 'fill': '#000', 'fill-opacity': .125, 'stroke': '#fff' });
  svg.style('font','10px sans-serif')

// .axis,
// .frame {
//   shape-rendering: crispEdges;
// }

// .axis line {
//   stroke: #ddd;
// }

// .axis path {
//   display: none;
// }

// .frame {
//   fill: none;
//   stroke: #aaa;
// }

// circle {
//   fill-opacity: .7;
// }

// circle.hidden {
//   fill: #ccc !important;
// }

// .extent {
//   fill: #000;
//   fill-opacity: .125;
//   stroke: #fff;
// }


}


