/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
 *
 * This program is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with this
 * program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **/

/**
 * Created by Dan on 5/24/2017.
 */



function wavelet_spectrogram_view(matrix_data,matrix_shape){
    var i0 = d3.interpolateHsvLong(d3.hsv(180, 1, 0.65), d3.hsv(90, 1, 0.90)),
    i1 = d3.interpolateHsvLong(d3.hsv(90, 1, 0.90), d3.hsv(0, 0, 0.95)),
    interpolateTerrain = function(t) { return t < 0.5 ? i0(t * 2) : i1((t - 0.5) * 2); },
    color = d3.scaleSequential(interpolateTerrain).domain([-10, 10]);
    var data = $.parseJSON(matrix_data);
    var dimensions = $.parseJSON(matrix_shape);
    var n=dimensions[0];
    var m=dimensions[1];
	var canvas = d3.select("canvas")
		  .attr("width", n)
		  .attr("height", m);

	  var context = canvas.node().getContext("2d"),
		  image = context.createImageData(n, m);
	  for (var j = 0, k = 0, l = 0; j < m*2; ++j) {
		for (var i = 0; i < n; ++i, ++k, l += 4) {
		  var c = d3.rgb(color(data[k]));
		  image.data[l + 0] = c.r;
		  image.data[l + 1] = c.g;
		  image.data[l + 2] = c.b;
		  image.data[l + 3] = 255;
		}
	  }

	  context.putImageData(image, 0, 0);
}
