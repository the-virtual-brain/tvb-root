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

function _VSI_bufferAtPoint(p, idx) {
    var result = HLPR_sphereBufferAtPoint(gl, p, 3, 12, 12);
    var bufferVertices= result[0];
    var bufferNormals = result[1];
    var bufferTriangles = result[2];
    var alphaAndColors = VSI_createColorBufferForSphere(false, idx, bufferVertices.numItems * 3);
    return [bufferVertices, bufferNormals, bufferTriangles, alphaAndColors[0], alphaAndColors[1]];
}

/**
 * Method used for creating a color buffer for a cube (measure point).
 *
 * @param isPicked If <code>true</code> then the color used will be
 * the one used for drawing the measure points for which the
 * corresponding eeg channels are selected.
 */
function VSI_createColorBufferForSphere(isPicked, nodeIdx, nrOfVertices) {
    var alphas = [];
    var alphaIndices = [];
    var pointAlphaIndex = [nodeIdx, 0, 0];

    for (var i = 0; i < nrOfVertices; i++) {
        alphaIndices = alphaIndices.concat(pointAlphaIndex);
        alphas = alphas.concat([1.0, 0.0]);
    }

    var alphaBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, alphaBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(alphas), gl.STATIC_DRAW);
    var alphaIndicesBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, alphaIndicesBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(alphaIndices), gl.STATIC_DRAW);
    return [alphaBuffer, alphaIndicesBuffer];
}


function _VSI_init_sphericalMeasurePoints(){
    for (var i = 0; i < NO_OF_MEASURE_POINTS; i++) {
        measurePointsBuffers[i] = _VSI_bufferAtPoint(measurePoints[i], i);
    }
}

function VSI_StartInternalSensorViewer(urlMeasurePoints,  noOfMeasurePoints, urlMeasurePointsLabels,
                                       shelfObject, minMeasure, maxMeasure, measure){
    _VS_static_entrypoint('', '[]', '', '', urlMeasurePoints, noOfMeasurePoints, '', '',
                         urlMeasurePointsLabels, '', shelfObject, null, false, false, true,
                         minMeasure, maxMeasure, measure);
    isInternalSensorView = true;
    displayMeasureNodes = true;

    _VSI_init_sphericalMeasurePoints();

}

function VSI_StartInternalActivityViewer(baseDatatypeURL, onePageSize, urlTimeList, urlVerticesList, urlLinesList,
                    urlTrianglesList, urlNormalsList, urlMeasurePoints, noOfMeasurePoints,
                    urlAlphasList, urlAlphasIndicesList, minActivity, maxActivity,
                    oneToOneMapping, doubleView, shelfObject, urlMeasurePointsLabels, boundaryURL) {

    _VS_movie_entrypoint(baseDatatypeURL, onePageSize, urlTimeList, urlVerticesList, urlLinesList,
                    urlTrianglesList, urlNormalsList, urlMeasurePoints, noOfMeasurePoints,
                    urlAlphasList, urlAlphasIndicesList, minActivity, maxActivity,
                    oneToOneMapping, doubleView, shelfObject, null, urlMeasurePointsLabels, boundaryURL);
    isInternalSensorView = true;
    displayMeasureNodes = true;
    isFaceToDisplay = true;

    _VSI_init_sphericalMeasurePoints();
}
