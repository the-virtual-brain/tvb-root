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

function _VSI_bufferAtPoint(p, idx) {
    const result = HLPR_sphereBufferAtPoint(gl, p, 3, 12, 12);
    const bufferVertices = result[0];
    const bufferNormals = result[1];
    const bufferTriangles = result[2];
    const vertexRegionBuffer = VSI_createColorBufferForSphere(idx, bufferVertices.numItems * 3);
    return [bufferVertices, bufferNormals, bufferTriangles, vertexRegionBuffer];
}

/**
 * Method used for creating a color buffer for a cube (measure point).
 */
function VSI_createColorBufferForSphere(nodeIdx, nrOfVertices) {
    let regionMap = [];
    for (let i = 0; i < nrOfVertices; i++) {
        regionMap.push(nodeIdx);
    }

    let vertexRegionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexRegionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(regionMap), gl.STATIC_DRAW);
    return vertexRegionBuffer;
}


function _VSI_init_sphericalMeasurePoints(){
    for (let i = 0; i < NO_OF_MEASURE_POINTS; i++) {
        measurePointsBuffers[i] = _VSI_bufferAtPoint(measurePoints[i], i);
    }
}

function VSI_StartInternalSensorViewer(urlMeasurePoints,  noOfMeasurePoints, urlMeasurePointsLabels,
                                       shellObject, minMeasure, maxMeasure, measure){
    _VS_static_entrypoint('', '[]', '', '', urlMeasurePoints, noOfMeasurePoints, '', urlMeasurePointsLabels, '',
                          shellObject, null, false, true, minMeasure, maxMeasure, measure);
    isInternalSensorView = true;
    displayMeasureNodes = true;

    _VSI_init_sphericalMeasurePoints();

}

function VSI_StartInternalActivityViewer(baseAdapterURL, onePageSize, urlTimeList, urlVerticesList, urlLinesList,
                    urlTrianglesList, urlNormalsList, urlMeasurePoints, noOfMeasurePoints,
                    urlRegionMapList, minActivity, maxActivity,
                    oneToOneMapping, doubleView, shellObject, urlMeasurePointsLabels, boundaryURL) {

    _VS_movie_entrypoint(baseAdapterURL, onePageSize, urlTimeList, urlVerticesList, urlLinesList,
                    urlTrianglesList, urlNormalsList, urlMeasurePoints, noOfMeasurePoints,
                    urlRegionMapList, minActivity, maxActivity,
                    oneToOneMapping, doubleView, shellObject, null, urlMeasurePointsLabels, boundaryURL);
    isInternalSensorView = true;
    displayMeasureNodes = true;
    isFaceToDisplay = true;

    _VSI_init_sphericalMeasurePoints();
}
