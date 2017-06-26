/**
 * Created by Dan on 5/24/2017.
 */



function wavelet_spectrogram_view(matrix_data,matrix_shape,vmin,vmax){
    var i0 = d3.interpolateHsvLong(d3.hsv(180, 1, 0.65), d3.hsv(90, 1, 0.90)),
    i1 = d3.interpolateHsvLong(d3.hsv(90, 1, 0.90), d3.hsv(0, 0, 0.95)),
    interpolateTerrain = function(t) { return t < 0.5 ? i0(t * 2) : i1((t - 0.5) * 2); },
    color = d3.scaleSequential(interpolateTerrain).domain([vmin, vmax]);
	ColSch_initColorSchemeComponent(vmin,vmax);
    var data = $.parseJSON(matrix_data);
    var dimensions = $.parseJSON(matrix_shape);
    var n=dimensions[0];
    var m=dimensions[1];
	var canvas = d3.select("canvas")
		  .attr("width", n)
		  .attr("height", m);

	  var context = canvas.node().getContext("2d"),
		  image = context.createImageData(m, n);
	  for (var j = 0, k = 0, l = 0; j < m; ++j) {
		for (var i = 0; i < n; ++i, ++k, l += 4) {
		  var c0 = d3.rgb(color(data[k]));
		  if(data[k]>vmax)
		  	data[k]=vmax;
		  if(data[k]<vmin)
		  	data[k]=vmin;
			var c=ColSch_getColor(data[k]);
		  image.data[l + 0] = c[0]*255;
		  image.data[l + 1] = c[1]*255;
		  image.data[l + 2] = c[2]*255;
		  image.data[l + 3] = 255;
		}
	  }
	image.className="rotate"+180;
	  context.putImageData(image, 0, 0);
}
