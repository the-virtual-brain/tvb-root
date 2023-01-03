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

// ----- Datatype methods mappings start from here
function setStateModeStep(stateVariable, mode, step){
    if (stateVariable === null || stateVariable === undefined) {
        stateVariable = 0;
    }
    if (mode === null || mode === undefined) {
        mode = 0;
    }
    if (step === null || step === undefined) {
        step = 1;
    }

    return [stateVariable, mode, step]
}

function readDataPageURL(baseDatatypeMethodURL, fromIdx, toIdx, stateVariable, mode, step) {
    const param_list = setStateModeStep(stateVariable, mode, step);
    return baseDatatypeMethodURL + '/read_data_page?from_idx=' + fromIdx + ";to_idx=" + toIdx + ";step=" + param_list[2] + ";specific_slices=[null," + param_list[0] + ",null," + param_list[1] + "]";
}

function readDataSplitPageURL(baseAdapterMethodURL, fromIdx, toIdx, stateVariable, mode, step) {
    const param_list = setStateModeStep(stateVariable, mode, step);
    return baseAdapterMethodURL+ "from_idx=" + fromIdx + ";to_idx=" + toIdx + ";step=" + param_list[2] + ";specific_slices=[null," + param_list[0] + ",null," + param_list[1] + "]";

}

function readDataChannelURL(baseDatatypeMethodURL, fromIdx, toIdx, stateVariable, mode, step, channels) {
    const baseURL = readDataPageURL(baseDatatypeMethodURL, fromIdx, toIdx, stateVariable, mode, step);
    return baseURL.replace('read_data_page', 'read_channels_page') + ';channels_list=' + channels;
}

// ------ Datatype methods mappings end here


// ------ Common movie data and functions

var AG_isStopped = false;

/**
 * Movie interaction
 */
function pauseMovie() {
    AG_isStopped = !AG_isStopped;
    const pauseButton = $("#ctrl-action-pause");
    if (AG_isStopped) {
        pauseButton.attr("class", "action action-controller-launch");
    } else {
        pauseButton.attr("class", "action action-controller-pause");
    }
}

// ------ Asynchronous geometry downloading begin
// These are used in the surface pick and tract views
//todo: A similar protocol might be useful for the rest of the 3d views: use doajaxcall, cancel all requests on error

/**
 * Initialize the buffers for the surface that should be displayed in non-pick mode.
 * callback is called when the buffers finished downloading
 * The callback receives an object {vertices: [...], indexes:[...], normals:[...]}
 */
function downloadBrainGeometry(urlVerticesDisplayList, urlTrianglesDisplayList,
                               urlNormalsDisplayList, callback) {

    const drawingBrain = {
        noOfUnloadedBuffers: 3,
        vertices: [], indexes: [], normals: [],
        success: callback
    };

    _startAsynchDownload(drawingBrain, $.parseJSON(urlVerticesDisplayList), drawingBrain.vertices);
    _startAsynchDownload(drawingBrain, $.parseJSON(urlNormalsDisplayList), drawingBrain.normals);
    _startAsynchDownload(drawingBrain, $.parseJSON(urlTrianglesDisplayList), drawingBrain.indexes);
}

function _startAsynchDownload(drawingBrain, urlList, results) {
    if (urlList.length === 0) {
        drawingBrain.noOfUnloadedBuffers -= 1;
        if (drawingBrain.noOfUnloadedBuffers === 0) {
            // Finished downloading buffer data.
            drawingBrain.success(drawingBrain);
        }
        return;
    }
    $.get(urlList[0], function (data) {
        results.push($.parseJSON(data));
        urlList.splice(0, 1);
        return _startAsynchDownload(drawingBrain, urlList, results);
    });
}

/**
 * Buffers the default gray surface color to the GPU
 * Saves the buffer object at index 3 & 4 in the buffersArray list
 */
function drawingBrainUploadDefaultColorBuffer(drawingBrainVertices, buffersArray) {
    for (let i = 0; i < drawingBrainVertices.length; i++) {
        const colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        const vertexCount = drawingBrainVertices[i].length / 3;
        const colors = new Float32Array(vertexCount * 4);
        gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
        buffersArray[i][4] = colorBuffer;

        const activityBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, activityBuffer);
        const activity = new Float32Array(vertexCount);
        gl.bufferData(gl.ARRAY_BUFFER, activity, gl.STATIC_DRAW);
        buffersArray[i][3] = activityBuffer;
    }
}

/**
 * downloadBrainGeometry callback recieves an object containig geometry description in js lists
 * This uploads that data to the gpu. It returns a list of 3 webglbuffers :[ vertex, normal, triangles]
 */
function drawingBrainUploadGeometryBuffers(drawingBrain) {
    let ret = [];
    for (let i = 0; i < drawingBrain.vertices.length; i++) {
        const vertexBuffer = HLPR_createWebGlBuffer(gl, drawingBrain.vertices[i], false, false);
        const normalBuffer = HLPR_createWebGlBuffer(gl, drawingBrain.normals[i], false, false);
        const trianglesBuffer = HLPR_createWebGlBuffer(gl, drawingBrain.indexes[i], true, false);
        ret.push([vertexBuffer, normalBuffer, trianglesBuffer]);
    }
    return ret;
}