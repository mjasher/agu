
function density(){

  d3.selectAll("#density svg").remove();

  var margin = {top: 20, right: 30, bottom: 30, left: 40},
      width = document.getElementById('density').clientWidth - margin.left - margin.right,
      height = 400 - margin.top - margin.bottom;

  var x = d3.scale.linear()
      .domain([30, 110])
      .range([0, width]);

  var y = d3.scale.linear()
      .domain([0, .1])
      .range([height, 0]);

  var xAxis = d3.svg.axis()
      .scale(x)
      .orient("bottom");

  var yAxis = d3.svg.axis()
      .scale(y)
      .orient("left")
      .tickFormat(d3.format("%"));

  var line = d3.svg.line()
      .x(function(d) { return x(d[0]); })
      .y(function(d) { return y(d[1]); });

  var histogram = d3.layout.histogram()
      .frequency(false)
      .bins(x.ticks(40));

  var svg = d3.select("#density").append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
    .append("text")
      .attr("class", "label")
      .attr("x", width)
      .attr("y", -6)
      .style("text-anchor", "end")
      .text("Output head (m)");

  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis);

  d3.json("data/density_demo.json", function(error, faithful) {
    var data = histogram(faithful),
        kde = kernelDensityEstimator(epanechnikovKernel(7), x.ticks(100));

    svg.selectAll(".bar")
        .data(data)
      .enter().insert("rect", ".axis")
        .attr("class", "bar")
        .attr("x", function(d) { return x(d.x) + 1; })
        .attr("y", function(d) { return y(d.y); })
        .attr("width", x(data[0].dx + data[0].x) - x(data[0].x) - 1)
        .attr("height", function(d) { return height - y(d.y); });

    svg.append("path")
        .datum(kde(faithful))
        .attr("class", "line")
        .attr("d", line);
  });

  function kernelDensityEstimator(kernel, x) {
    return function(sample) {
      return x.map(function(x) {
        return [x, d3.mean(sample, function(v) { return kernel(x - v); })];
      });
    };
  }

  function epanechnikovKernel(scale) {
    return function(u) {
      return Math.abs(u /= scale) <= 1 ? .75 * (1 - u * u) / scale : 0;
    };
  }

}

