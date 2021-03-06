

function grouped_bar(el, data){


  var margin = {top: 20, right: 20, bottom: 60, left: 40},
      width = el.node().clientWidth - margin.left - margin.right,
      // height = 400 - margin.top - margin.bottom;
      height = width - margin.top - margin.bottom;

  var x0 = d3.scale.ordinal()
      .rangeRoundBands([0, width], .1);

  var x1 = d3.scale.ordinal();

  var y = d3.scale.linear()
      .range([height, 0]);

  var color = d3.scale.ordinal()
      // .range(["#98abc5", "#8a89a6", "#7b6888", "#6b486b", "#a05d56", "#d0743c", "#ff8c00"]);
      // .range(["#1f77b4",  "#2ca02c"]);
      .range(["steelblue",  "darkred"]);
  // var color = d3.scale.category10();

  var xAxis = d3.svg.axis()
      .scale(x0)
      .orient("bottom");

  var yAxis = d3.svg.axis()
      .scale(y)
      .orient("left")
      .tickFormat(d3.format("%"));

  var svg = el.append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var ageNames = d3.keys(data.data[0]).filter(function(key) { return key !== "State"; });

    data.data.forEach(function(d) {
      d.ages = ageNames.map(function(name) { return {name: name, value: +d[name]}; });
    });

    x0.domain(data.data.map(function(d) { return d.State; }));
    x1.domain(ageNames).rangeRoundBands([0, x0.rangeBand()]);
    y.domain([0, d3.max(data.data, function(d) { return d3.max(d.ages, function(d) { return d.value; }); })]);


    var state = svg.selectAll(".state")
        .data(data.data)
      .enter().append("g")
        .attr("class", "g")
        .attr("transform", function(d) { return "translate(" + x0(d.State) + ",0)"; });

    state.selectAll("rect")
        .data(function(d) { return d.ages; })
      .enter().append("rect")
        .attr("width", x1.rangeBand())
        .attr("x", function(d) { return x1(d.name); })
        .attr("y", function(d) { return y(d.value); })
        .attr("height", function(d) { return height - y(d.value); })
        .style("fill", function(d) { return color(d.name); });

    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis)
        .selectAll('text')
        .style("text-anchor", "end")
        .attr("transform", "translate(0,-5)rotate(-45)");

    svg.append("g")
        .attr("class", "y axis")
        .call(yAxis)
      .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", ".71em")
        .style("text-anchor", "end")
        .text("Fraction of variance");

  svg.append('text').text(data.labels.title).style("text-anchor", "middle").style('font','14px sans-serif').attr("transform", "translate(" + (width/2) + "," + 15 + ")");


    // var legend = svg.selectAll(".legend")
    //     .data(ageNames.slice().reverse())
    //   .enter().append("g")
    //     .attr("class", "legend")
    //     .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

    // legend.append("rect")
    //     .attr("x", width - 18)
    //     .attr("width", 18)
    //     .attr("height", 18)
    //     .style("fill", color);

    // legend.append("text")
    //     .attr("x", width - 24)
    //     .attr("y", 9)
    //     .attr("dy", ".35em")
    //     .style("text-anchor", "end")
    //     .text(function(d) { return d; });

 
  svg.selectAll('.axis line').style({'stroke': '#000', 'fill': 'none', 'shape-rendering': 'crispEdges'});
  svg.selectAll('.axis path').style({'stroke': '#000', 'fill': 'none', 'shape-rendering': 'crispEdges', 'display': 'none'});
  // svg.selectAll('.axis path').style('display', 'none');
  svg.style('font','10px sans-serif');


// .axis path,
// .axis line {
//   fill: none;
//   stroke: #000;
//   shape-rendering: crispEdges;
// }

// .bar {
//   fill: steelblue;
// }

// .x.axis path {
//   display: none;
// }

// svg{
//   font: 10px sans-serif;
// }





}
