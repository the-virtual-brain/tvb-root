/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and 
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
 *
 * This program is free software; you can redistribute it and/or modify it under 
 * the terms of the GNU General Public License version 2 as published by the Free
 * Software Foundation. This program is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
 * License for more details. You should have received a copy of the GNU General 
 * Public License along with this program; if not, you can download it here
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0
 *
 **/

////////////////////////////////////~~~~~~~~~~~~START LEGEND RELATED CODE~~~~~~~~~~~~~~~~///////////////////////////
var legendXMin = 80;
var legendXMax = 83;
var legendYMin = -63;
var legendYMax = 63;
var legendZ = -150;
var legendGranularity = 127;
var legend_activity_values = [];

///// We draw legend with WebGL, so for the legend object we need buffers.
var LEG_legendBuffers = [];

var legendMin = 0;
var legendMax = 1;

function LEG_initMinMax(minVal, maxVal) {
	legendMin = minVal;
	legendMax = maxVal;
    ColSch_initColorSchemeParams(minVal, maxVal, LEG_updateLegendColors);
}

function LEG_updateLegendXMinAndXMax() {
    //800/600 = 1.33
    legendXMin = (gl.viewportWidth/gl.viewportHeight) * 78/1.33;
    legendXMax = (gl.viewportWidth/gl.viewportHeight) * 78/1.33 + 3;
}

function LEG_updateLegendVerticesBuffers() {
    LEG_updateLegendXMinAndXMax();

    var vertices = [];
    var inc = (legendYMax - legendYMin) / legendGranularity;
    for (var i = legendYMin; i <= legendYMax; i = i + inc) {
        vertices = vertices.concat([legendXMax, i, legendZ]);
        vertices = vertices.concat([legendXMin, i, legendZ]);
    }

    LEG_legendBuffers[0] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, LEG_legendBuffers[0]);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    LEG_legendBuffers[0].itemSize = 3;
    LEG_legendBuffers[0].numItems = vertices.length / 3;
}

function LEG_generateLegendBuffers() {
    LEG_updateLegendXMinAndXMax();

	var vertices = [];
	var normals = [];
	var indices = [];

	var inc = (legendYMax - legendYMin) / legendGranularity;
	var activityDiff = legendMax - legendMin;
    legend_activity_values = [];        // empty the set, or the gradient will get higher on subsequent calls
	for (var i=legendYMin; i<=legendYMax; i=i+inc) {
		vertices = vertices.concat([legendXMax, i, legendZ]);
		vertices = vertices.concat([legendXMin, i, legendZ]);
		normals = normals.concat([0, 0, 1, 0, 0, 1]);
		var activityValue = legendMin + activityDiff * ((i - legendYMin) / (legendYMax - legendYMin));
		legend_activity_values = legend_activity_values.concat([activityValue, activityValue]);
	}
	for (var i=0; i<vertices.length/3 - 2; i++) {
		indices = indices.concat([i, i+1, i+2]);
	}
	LEG_legendBuffers[0] = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, LEG_legendBuffers[0]);
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
	LEG_legendBuffers[0].itemSize = 3;
	LEG_legendBuffers[0].numItems = vertices.length / 3;
	
	LEG_legendBuffers[1] = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, LEG_legendBuffers[1]);
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
	LEG_legendBuffers[1].itemSize = 3;
	LEG_legendBuffers[1].numItems = normals.length / 3;
	
	LEG_legendBuffers[2] = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, LEG_legendBuffers[2]);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    LEG_legendBuffers[2].itemSize = 1;
    LEG_legendBuffers[2].numItems = indices.length;
    
    if (isOneToOneMapping) {
    	var colors = []
        for (var i=0; i < LEG_legendBuffers[0].numItems* 4; i++) {
        	colors = colors.concat(0, 0, 1, 1.0);
        }
    	LEG_legendBuffers[3] = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, LEG_legendBuffers[3]);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
        
    } else {
    	var alphas = [];
    	var alphasIndices = [];
    	for (var i=0; i<legend_activity_values.length/2; i++) {
    		alphas = alphas.concat([1.0, 0.0, 1.0, 0.0]);
    		alphasIndices = alphasIndices.concat([i + NO_OF_MEASURE_POINTS + 2, 1, 1, i + NO_OF_MEASURE_POINTS + 2, 1, 1])
    	}
    	
    	LEG_legendBuffers[3] = gl.createBuffer();
	    gl.bindBuffer(gl.ARRAY_BUFFER, LEG_legendBuffers[3]);
	    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(alphas), gl.STATIC_DRAW);
	    
	    LEG_legendBuffers[4] = gl.createBuffer();
	    gl.bindBuffer(gl.ARRAY_BUFFER, LEG_legendBuffers[4]);
	    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(alphasIndices), gl.STATIC_DRAW);    
	}	
    LEG_updateLegendColors()
}

/**
 * Refresh color buffer for legend.
 */
function LEG_updateLegendColors() {
    if (isOneToOneMapping) {
    	var upperBorder = legend_activity_values.length
    	var colors = new Float32Array(upperBorder * 4)
        getGradientColorArray(legend_activity_values, legendMin, legendMax, colors)
        LEG_legendBuffers[3] = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, LEG_legendBuffers[3]);
        gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
    }
    else
        for (var i = 0; i < legend_activity_values.length / 2; i++) {
            var idx = i + NO_OF_MEASURE_POINTS + 2
            var rgb = getGradientColor(legend_activity_values[i * 2], legendMin, legendMax)
            gl.uniform4f(shaderProgram.colorsUniform[idx], rgb[0], rgb[1], rgb[2], 1);
    }
}
/////////////////////////////////////////~~~~~~~~END LEGEND RELATED CODE~~~~~~~~~~~//////////////////////////////////


