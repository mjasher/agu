

function uncertainty(){
    
    d3.selectAll("#uncertainty svg").remove();

    var margin = {top: 100, right: 100, bottom: 100, left: 100},
        width = document.getElementById("uncertainty").clientWidth - margin.left - margin.right,
        height = document.getElementById("uncertainty").clientWidth - margin.top - margin.bottom;

    var x = d3.scale.ordinal()
        .domain(["statistical", "scenario", "recognised ignorance"])
        .rangePoints([0, width]);

    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("bottom");

    var y = d3.scale.ordinal()
        .domain(["context", "model", "inputs", "parameters"])
        .rangePoints([0, height]);

    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left");

    var z = d3.scale.ordinal()
        .domain(["context", "model", "inputs", "parameters"])
        .rangePoints([0, height]);

    var zAxis = d3.svg.axis()
        .scale(z)
        .orient("left");

    var svg = d3.select("#uncertainty").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    svg.append("g")
        .attr("class", "x axis")
        .call(xAxis)
        .attr('transform', 'translate(0,'+height+')');

    svg.append("g")
        .attr("class", "y axis")
        .call(yAxis);

    svg.append("g")
        .attr("class", "z axis")
        .call(zAxis)
        .attr('transform', 'rotate(45)');


}

