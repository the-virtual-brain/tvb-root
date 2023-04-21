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

/*
 * This file contains helper functions that are used mainly by the connectivity and the connectivity 3D
 * WebGL based visualizers. 
 */

/*
 * Given an input file containing the center positions and the labels, read these data
 * and compute the required normalization steps in order to center the brain on the canvas.
 */
function HLPR_readPointsAndLabels(filePoints, urlLabels) {
    var positions = HLPR_readJSONfromFile(filePoints);
    var labels = HLPR_readJSONfromFile(urlLabels);
	var steps = HLPR_computeNormalizationSteps(positions);
    return [positions, labels, steps[0], steps[1]];
}

function HLPR_computeNormalizationSteps(positions) {
	/*
	 * Compute normalization steps in order to center the connectivity matrix in 0,0
	 */
    var minX = positions[0][0];
    var maxX = positions[0][0];
    var minY = positions[0][1];
    var maxY = positions[0][1];

    for (var i = 0; i < positions.length; i++) {
        if (positions[i][0] < minX) {
            minX = positions[i][0];
        }
        if (positions[i][0] > maxX) {
            maxX = positions[i][0];
        }
        if (positions[i][1] < minY) {
            minY = positions[i][1];
        }
        if (positions[i][1] > maxY) {
            maxY = positions[i][1];
        }
    }
    var stepX = (minX + maxX) / 2.0;
    var stepY = (minY + maxY) / 2.0;
    return [stepX, stepY];
}

/*
 * Helper function, remove a given element from an array. We need this for example when cutting lines.
 */
function HLPR_removeByElement(arrayName, arrayElement) {
  for(var i=0; i<arrayName.length; i++ ) { 
	  if(arrayName[i]==arrayElement)  {
          arrayName.splice(i, 1);
      }
  }
}

function HLPR_sphereBufferAtPoint(gl, point, radius, latitudeBands, longitudeBands) {
    var moonVertexPositionBuffer;
    var moonVertexNormalBuffer;
    var moonVertexIndexBuffer;

    latitudeBands = latitudeBands || 30;
    longitudeBands = longitudeBands || 30;

    var vertexPositionData = [];
    var normalData = [];
    for (var latNumber = 0; latNumber <= latitudeBands; latNumber++) {
        var theta = latNumber * Math.PI / latitudeBands;
        var sinTheta = Math.sin(theta);
        var cosTheta = Math.cos(theta);

        for (var longNumber = 0; longNumber <= longitudeBands; longNumber++) {
            var phi = longNumber * 2 * Math.PI / longitudeBands;
            var sinPhi = Math.sin(phi);
            var cosPhi = Math.cos(phi);

            var x = cosPhi * sinTheta;
            var y = cosTheta;
            var z = sinPhi * sinTheta;

            normalData.push(x);
            normalData.push(y);
            normalData.push(z);
            vertexPositionData.push(parseFloat(point[0]) + radius * x);
            vertexPositionData.push(parseFloat(point[1]) + radius * y);
            vertexPositionData.push(parseFloat(point[2]) + radius * z);
        }
    }

    var indexData = [];
    for (latNumber = 0; latNumber < latitudeBands; latNumber++) {
        for (longNumber = 0; longNumber < longitudeBands; longNumber++) {
            var first = (latNumber * (longitudeBands + 1)) + longNumber;
            var second = first + longitudeBands + 1;
            indexData.push(first);
            indexData.push(second);
            indexData.push(first + 1);

            indexData.push(second);
            indexData.push(second + 1);
            indexData.push(first + 1);
        }
    }

    moonVertexNormalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, moonVertexNormalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normalData), gl.STATIC_DRAW);
    moonVertexNormalBuffer.itemSize = 3;
    moonVertexNormalBuffer.numItems = normalData.length / 3;

    moonVertexPositionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, moonVertexPositionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertexPositionData), gl.STATIC_DRAW);
    moonVertexPositionBuffer.itemSize = 3;
    moonVertexPositionBuffer.numItems = vertexPositionData.length / 3;

    moonVertexIndexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, moonVertexIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indexData), gl.STATIC_DRAW);
    moonVertexIndexBuffer.itemSize = 1;
    moonVertexIndexBuffer.numItems = indexData.length;

    return [moonVertexPositionBuffer, moonVertexNormalBuffer, moonVertexIndexBuffer];
}


/**
 * Create vertices, normals and triangles buffers for a cube, around point p.
 * @param {Object} p center of the cube. (expected [x, y, z] )
 */
function HLPR_bufferAtPoint(glcontext, p) {
    var bufferVertices = glcontext.createBuffer();
    glcontext.bindBuffer(glcontext.ARRAY_BUFFER, bufferVertices);
    var PS = 3.0;
    var vertices = [p[0] - PS, p[1] - PS, p[2] + PS, // Front face
			        p[0] + PS, p[1] - PS, p[2] + PS,
			        p[0] + PS, p[1] + PS, p[2] + PS,
			        p[0] - PS, p[1] + PS, p[2] + PS,
			        p[0] - PS, p[1] - PS, p[2] - PS, // Back face
			        p[0] - PS, p[1] + PS, p[2] - PS,
			        p[0] + PS, p[1] + PS, p[2] - PS,
			        p[0] + PS, p[1] - PS, p[2] - PS,
			        p[0] - PS, p[1] + PS, p[2] - PS, // Top face
			        p[0] - PS, p[1] + PS, p[2] + PS,
			        p[0] + PS, p[1] + PS, p[2] + PS,
			        p[0] + PS, p[1] + PS, p[2] - PS,
			        p[0] - PS, p[1] - PS, p[2] - PS,  // Bottom face
			        p[0] + PS, p[1] - PS, p[2] - PS,
			        p[0] + PS, p[1] - PS, p[2] + PS,
			        p[0] - PS, p[1] - PS, p[2] + PS,
			        p[0] + PS, p[1] - PS, p[2] - PS,  // Right face
			        p[0] + PS, p[1] + PS, p[2] - PS,
			        p[0] + PS, p[1] + PS, p[2] + PS,
			        p[0] + PS, p[1] - PS, p[2] + PS,
			        p[0] - PS, p[1] - PS, p[2] - PS,  // Left face
			        p[0] - PS, p[1] - PS, p[2] + PS,
			        p[0] - PS, p[1] + PS, p[2] + PS,
			        p[0] - PS, p[1] + PS, p[2] - PS];
    glcontext.bufferData(glcontext.ARRAY_BUFFER, new Float32Array(vertices), glcontext.STATIC_DRAW);
    bufferVertices.itemSize = 3;

    var bufferNormals = glcontext.createBuffer();
    glcontext.bindBuffer(glcontext.ARRAY_BUFFER, bufferNormals);
    var normals = [ 0.0,  0.0,  1.0,  // Front
                    0.0,  0.0,  1.0,
                    0.0,  0.0,  1.0,
                    0.0,  0.0,  1.0,
                    0.0,  0.0, -1.0,  // Back
                    0.0,  0.0, -1.0,
                    0.0,  0.0, -1.0,
                    0.0,  0.0, -1.0,
                    0.0,  1.0,  0.0,  // Top
                    0.0,  1.0,  0.0,
                    0.0,  1.0,  0.0,
                    0.0,  1.0,  0.0,
                    0.0, -1.0,  0.0,  // Bottom
                    0.0, -1.0,  0.0,
                    0.0, -1.0,  0.0,
                    0.0, -1.0,  0.0,
                    1.0,  0.0,  0.0,  // Right
                    1.0,  0.0,  0.0,
                    1.0,  0.0,  0.0,
                    1.0,  0.0,  0.0,
                   -1.0,  0.0,  0.0,  // Left
                   -1.0,  0.0,  0.0,
                   -1.0,  0.0,  0.0,
                   -1.0,  0.0,  0.0];
    glcontext.bufferData(glcontext.ARRAY_BUFFER, new Float32Array(normals), glcontext.STATIC_DRAW);
    bufferNormals.itemSize = 3;

    var bufferTriangles = glcontext.createBuffer();
    glcontext.bindBuffer(glcontext.ELEMENT_ARRAY_BUFFER, bufferTriangles);
    var triangs = [ 0,  1,  2,      0,  2,  3,    // front
			        4,  5,  6,      4,  6,  7,    // back
			        8,  9,  10,     8,  10, 11,   // top
			        12, 13, 14,     12, 14, 15,   // bottom
			        16, 17, 18,     16, 18, 19,   // right
			        20, 21, 22,     20, 22, 23    // left
			      ];
    glcontext.bufferData(glcontext.ELEMENT_ARRAY_BUFFER, new Uint16Array(triangs), glcontext.STATIC_DRAW);
    bufferTriangles.numItems = 36;
    
    return [bufferVertices, bufferNormals, bufferTriangles];
}


/**
 * Create webgl buffers from the specified files
 *
 * @param dataList the list of JS data
 * @return a list in which will be added the buffers created based on the data from the specified files
 */
function HLPR_getDataBuffers(glcontext, data_url_list, staticFiles, isIndex) {
    var result = [];
    for (var i = 0; i < data_url_list.length; i++) {
        var data_json = HLPR_readJSONfromFile(data_url_list[i], staticFiles);
        var buffer = HLPR_createWebGlBuffer(glcontext, data_json, isIndex, staticFiles);
        result.push(buffer);
        data_json = null;
    }
    return result;
}


/**
 * Creates a web gl buffer from the given data.
 *
 * @param glcontext the webgl contex
 * @param data_json the list of data
 * @param isIndex true if the method should create an index buffer
 * @param staticFile
 */
function HLPR_createWebGlBuffer(glcontext, data_json, isIndex, staticFile) {
    var buffer = glcontext.createBuffer();
    if (isIndex) {
        if (staticFile) {
            for (var j = 0; j < data_json.length; j++) {
                data_json[j] = parseInt(data_json[j]);
            }
        }
        glcontext.bindBuffer(glcontext.ELEMENT_ARRAY_BUFFER, buffer);
        glcontext.bufferData(glcontext.ELEMENT_ARRAY_BUFFER, new Uint16Array(data_json), glcontext.STATIC_DRAW);
    } else {
        if (staticFile) {
            for (var j = 0; j < data_json.length; j++) {
                data_json[j] = parseFloat(data_json[j]);
            }
        }
        glcontext.bindBuffer(glcontext.ARRAY_BUFFER, buffer);
        glcontext.bufferData(glcontext.ARRAY_BUFFER, new Float32Array(data_json), glcontext.STATIC_DRAW);
    }
    buffer.numItems = data_json.length;
    return buffer;
}
