/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need to download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

/* globals gl, NO_OF_MEASURE_POINTS*/
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
        LEG_legendBuffers[3] = gl.createBuffer();
        LEG_legendBuffers[3].numItems = LEG_legendBuffers[0].numItems;
        gl.bindBuffer(gl.ARRAY_BUFFER, LEG_legendBuffers[3]);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(legend_activity_values), gl.STATIC_DRAW);

        // This buffer is a color buffer. It is used by the surface pick vertex shader.
        LEG_legendBuffers[4] = gl.createBuffer();
        LEG_legendBuffers[4].numItems = LEG_legendBuffers[0].numItems*4;
        gl.bindBuffer(gl.ARRAY_BUFFER, LEG_legendBuffers[4]);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(LEG_legendBuffers[0].numItems*4), gl.STATIC_DRAW);
    } else {
        var vertexRegion = [];
        for (var i=0; i<legend_activity_values.length/2; i++) {
            vertexRegion = vertexRegion.concat([i + NO_OF_MEASURE_POINTS + 2, i + NO_OF_MEASURE_POINTS + 2]);
        }

        LEG_legendBuffers[3] = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, LEG_legendBuffers[3]);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertexRegion), gl.STATIC_DRAW);
    }
    LEG_updateLegendColors();
}

/**
 * Refresh color uniforms for legend. If isOneToOneMapping it does nothing.
 */
function LEG_updateLegendColors() {
    if (!isOneToOneMapping) {
        var col = ColSchInfo();
        for (var i = 0; i < legend_activity_values.length / 2; i++) {
            var idx = i + NO_OF_MEASURE_POINTS + 2;
            gl.uniform2f(GL_shaderProgram.activityUniform[idx], legend_activity_values[i * 2], col.tex_v);
        }
    }
}
/////////////////////////////////////////~~~~~~~~END LEGEND RELATED CODE~~~~~~~~~~~//////////////////////////////////


