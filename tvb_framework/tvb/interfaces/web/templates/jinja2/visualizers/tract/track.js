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
 * .. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
 **/

var near = 0.1;

var tracts_element_buffers;
var tract_vbuffers;
var chunks_tract_offsets;

var tracts_loaded = false;

var TRACK_shellDisplayBuffers = [];

var drawingMode;

var BRAIN_CANVAS_ID = "GLcanvas";

var isFaceToDisplay = true;


/* globals gl, GL_shaderProgram, defaultLightSettings, GL_colorPickerInitColors,
           updateGLCanvasSize, LEG_updateLegendVerticesBuffers, initGL, basicInitShaders,
           ColSchGetTheme, displayMessage, perspective, mvRotate, mvTranslate, mvPushMatrix, mvPopMatrix,
           basicSetLighting, HLPR_createWebGlBuffer, GL_handleMouseUp, createAndUseShader
           */

function _customInitGL(canvas) {
    window.onresize = function() {
        updateGLCanvasSize(BRAIN_CANVAS_ID);
    };
    initGL(canvas);
    canvas.redrawFunctionRef = drawScene;
    drawingMode = gl.TRIANGLES;
}


function _initShaders() {
    createAndUseShader("shader-fs", "shader-vs");
    SHADING_Context.surface_pick_init(GL_shaderProgram);
    gl.uniform1i(GL_shaderProgram.useVertexColors, true);
}

function getTractElementBuffers(url){
    chunks_tract_offsets = HLPR_readJSONfromFile(url);
    var buffers = [];
    // reconstruct tract element buffers from the offsets
    for (var cnk = 0 ; cnk < chunks_tract_offsets.length; cnk++) {
        var tract_offsets = chunks_tract_offsets[cnk];
        var line_el = [];

        for (var tid = 0; tid < tract_offsets.length - 1; tid++) {
            var len = tract_offsets[tid + 1] - tract_offsets[tid];

            line_el.push(tract_offsets[tid]);
            for (var i = 1; i < len - 1; i++) {
                line_el.push(tract_offsets[tid] + i);
                line_el.push(tract_offsets[tid] + i);
            }
            line_el.push(tract_offsets[tid] + len - 1);
        }
        buffers.push(HLPR_createWebGlBuffer(gl, line_el, true, false));
    }
    return buffers;
}

function vertices_to_directions(tract_lines, tract_offsets){
    function normalized_mean_direction(buff, n, p){
        var mean_dir = [];
        mean_dir[0] = buff[3*n] - buff[3*p]; // last_track_vtx.x - first_track_vtx.x
        mean_dir[1] = buff[3*n+1] - buff[3*p+1];
        mean_dir[2] = buff[3*n+2] - buff[3*p+2];
        var norm = Math.sqrt(mean_dir[0]*mean_dir[0] + mean_dir[1]*mean_dir[1] + mean_dir[2]*mean_dir[2]);
        mean_dir[0] = mean_dir[0]/norm;
        mean_dir[1] = mean_dir[1]/norm;
        mean_dir[2] = mean_dir[2]/norm;
        return mean_dir;
    }

    var colors = new Float32Array(tract_lines.buffer.length/3*4);
    var buff = tract_lines.buffer;

    // one color per track, based on mean direction of track
    for (var tid = 0; tid < tract_offsets.length - 1; tid++) {
        var n = tract_offsets[tid + 1] - 1;
        var p = tract_offsets[tid];
        var mean_direction = normalized_mean_direction(buff, n, p);

        for(var i = 4*p; i < 4*n; i+=4){
            colors[i] = Math.abs(mean_direction[0]);
            colors[i+1] = Math.abs(mean_direction[1]);
            colors[i+2] = Math.abs(mean_direction[2]);
            colors[i+3] = 1.0;
        }
    }

    // direction of last vertex is not well defined, make it tha same as it's predecessor's
    var lvi = tract_lines.shape[0]-1;
    colors[lvi] = colors[lvi-4];
    colors[lvi-1] = colors[lvi-5];
    colors[lvi-2] = colors[lvi-6];
    colors[lvi-3] = colors[lvi-7];
    return colors;
}


function fetch_tracts_for_region(urlTrackStarts, urlTrackVertices, region){
    tracts_element_buffers = getTractElementBuffers(urlTrackStarts + '?region_id='+region);
    tract_vbuffers = [];
    tracts_loaded = false;

    if(tracts_element_buffers.length === 0){
        displayMessage('no tracks for selected region', 'warningMessage');
        return;
    }

    var latch_count = tracts_element_buffers.length;
    function _vertices_recv(tract_lines, idx){
        var pos = HLPR_createWebGlBuffer(gl, tract_lines.buffer, false, false);
        var colors = vertices_to_directions(tract_lines, chunks_tract_offsets[idx]);
        colors = HLPR_createWebGlBuffer(gl, colors, false, false);

        tract_vbuffers[idx] = [pos, pos, pos, colors, colors];//pos, elem, norm, color, activ
        latch_count -= 1;
        if (latch_count === 0){
            tracts_loaded = true;
            displayMessage('Tracts loaded', 'infoMessage');
            closeOverlay();
            drawScene();
        }
    }

    showOverlay("/showBlockerOverlay", false, 'message_data=Downloading tracts');
    displayMessage('Downloading tracts', 'warningMessage');

    for (var i=0; i < tracts_element_buffers.length; i++) {
        var url = urlTrackVertices + '?region_id=' + region + '&slice_number=' + i;
        HLPR_fetchNdArray(url, _vertices_recv, i);
    }
}

function drawTractLines(linesBuffers, brainBuffers, bufferSetsMask) {
    var lightSettings = basicSetLighting({
        ambientColor: [0.9, 0.9, 0.9],
        directionalColor : [0.05, 0.05, 0.05],
        specularColor : [0.0, 0.0, 0.0],
        materialColor: [1.0, 1.0, 1.0, 1.0]
    });

    gl.uniform1i(GL_shaderProgram.useActivity, false);
    gl.uniform1i(GL_shaderProgram.useVertexColors, true);

    gl.lineWidth(1.0);
    // we want all the brain buffers in this set except the element array buffer (at index 2)
    var bufferSets = [];
    for (var c = 0; c < brainBuffers.length; c++){
        var chunk = brainBuffers[c].slice();
        chunk[2] = linesBuffers[c];
        bufferSets.push(chunk);
    }
    drawBuffers(gl.LINES, bufferSets, bufferSetsMask);
    basicSetLighting(lightSettings);
}

function _bindEvents(canvas){
    // Enable keyboard and mouse interaction
    canvas.onkeydown = function(event){
        GL_handleKeyDown(event);
        drawScene();
    };
    canvas.onkeyup = function(event){
        GL_handleKeyUp(event);
        drawScene();
    };
    canvas.onmousedown = function(event){
        GL_handleMouseDown(event, $("#" + BRAIN_CANVAS_ID));
    };
    canvas.oncontextmenu = function(){return false;};
    $(document).on('mouseup', function(event){
        GL_handleMouseUp(event);
        drawScene();
    });
    $(document).on('mousemove', function(event){
        GL_handleMouseMove(event);
        drawScene();
    });
    $(canvas).mousewheel(function(event, delta) {
        GL_handleMouseWeel(delta);
        drawScene();
        return false; // prevent default
    });
}

function TRACK_webGLStart(urlTrackStarts, urlTrackVertices, shellObject) {
    var canvas = document.getElementById(BRAIN_CANVAS_ID);
    shellObject = $.parseJSON(shellObject);

    _customInitGL(canvas);
    _initShaders();

    displayMessage("Start loading surface data!", "infoMessage");
    downloadBrainGeometry($.toJSON(shellObject[0]), $.toJSON(shellObject[2]), $.toJSON(shellObject[1]),
        function(drawingBrain){
            displayMessage("Finished loading surface data!", "infoMessage");
            // Finished downloading buffer data. Initialize TRACK_shellDisplayBuffers
            TRACK_shellDisplayBuffers = drawingBrainUploadGeometryBuffers(drawingBrain);
            drawingBrainUploadDefaultColorBuffer(drawingBrain.vertices, TRACK_shellDisplayBuffers);
            drawScene();
        }
    );

    ColSch_initColorSchemeComponent();
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    _bindEvents(canvas);

    drawScene();

    var vs_regionsSelector = new TVBUI.RegionSelectComponent("#channelSelector", {boxesSelector:'input[type=radio]'});

    vs_regionsSelector.change(function(value){
        var regionId = parseInt(value[0], 10);
        fetch_tracts_for_region(urlTrackStarts, urlTrackVertices, regionId);
    });

    vs_regionsSelector.val([2]);
}


/**
 * Redraw from buffers.
 */
function drawScene() {
    if (TRACK_shellDisplayBuffers.length === 0) {
        displayMessage("The load operation for the surface data is not completed yet!", "infoMessage");
        return;
    }

    gl.enable(gl.BLEND);
    gl.enable(gl.DITHER);
    basicSetLighting();

    var theme = ColSchGetTheme().surfaceViewer;
    gl.clearColor(theme.backgroundColor[0], theme.backgroundColor[1], theme.backgroundColor[2], theme.backgroundColor[3]);
    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    // View angle is 45, we want to see object from 0.1 up to 800 distance from viewer
    var aspect = gl.viewportWidth / gl.viewportHeight;
    perspective(45, aspect , near, 800.0);

    mvPushMatrix();
    mvRotate(180, [0, 0, 1]);

    if(tracts_loaded){
        gl.uniform1i(GL_shaderProgram.useActivity, false);
        drawTractLines(tracts_element_buffers, tract_vbuffers );
    }

    if (isFaceToDisplay) {
        gl.uniform1i(GL_shaderProgram.useActivity, true);
        drawBuffers(drawingMode, TRACK_shellDisplayBuffers, true, gl.FRONT);
    }

    mvPopMatrix();
}


function drawBuffers(drawMode, buffersSets,useBlending, cullFace) {
    if (useBlending) {
        var lightSettings = setLighting(blendingLightSettings);
        gl.enable(gl.BLEND);
        gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);
        gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        // Blending function for alpha: transparent pix blended over opaque -> opaque pix
        if (cullFace) {
            gl.enable(gl.CULL_FACE);
            gl.cullFace(cullFace);
        }
    }

    setMatrixUniforms();
    for (var i = 0; i < buffersSets.length; i++) {
        SHADING_Context.surface_pick_draw(GL_shaderProgram,
            buffersSets[i][0], buffersSets[i][1],
            buffersSets[i][4], buffersSets[i][3],
            buffersSets[i][2], drawMode
        );
    }
    if (useBlending) {
        gl.disable(gl.BLEND);
        gl.disable(gl.CULL_FACE);
        setLighting(lightSettings);
        // Draw the same transparent object the second time
        if (cullFace === gl.FRONT) {
            drawBuffers(drawMode, buffersSets, useBlending, gl.BACK);
        }
    }
}

function switchFaceObject() {
    isFaceToDisplay = !isFaceToDisplay;
    drawScene();
}