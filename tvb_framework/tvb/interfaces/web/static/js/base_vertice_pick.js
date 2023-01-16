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


let BASE_PICK_doPick = false;
var near = 0.1;

// Absolute triangle index in the Surface (it is compatible with the original surface.triangles attribute)
let TRIANGLE_pickedIndex = -1;
let VERTEX_pickedIndex = -1;

// Coordinates for the selection 'pins'
let navigatorX = 0.0, navigatorY = 0.0, navigatorZ = 0.0;
let navigatorXrot = 0.0, navigatorYrot = 0.0;

let drawingBrainVertices = []; // kept only for the vertex counts

let BASE_PICK_brainPickingBuffers = [];
let BASE_PICK_brainDisplayBuffers = [];
const BASE_PICK_navigatorBuffers = [];

//If we are in movie mode stop the custom redraw on all events since it will redraw automatically anyway.
let BASE_PICK_isMovieMode = false;
//A dictionary that hold information needed to draw the surface focal points
let surfaceFocalPoints = {};

var drawingMode;
let BASE_PICK_regionBoundariesController = null;

var BRAIN_CANVAS_ID = "GLcanvas";
let BASE_PICK_pinBuffers = [];
let BRAIN_CENTER = null;

//Keep the vertices data for it will be needed later for the actual coordinates
//of the selected point and for computing the closest vertices when using '2d-type-selection'
//These are the picking brain's vertices
let verticesPoints = [];

let picking_triangles_number = [];

/* globals gl, GL_shaderProgram, isOneToOneMapping, defaultLightSettings, GL_colorPickerInitColors,
 updateGLCanvasSize, LEG_updateLegendVerticesBuffers, initGL, basicInitShaders,
 ColSchGetTheme, displayMessage, perspective, mvRotate, mvTranslate, mvPushMatrix, mvPopMatrix,
 basicSetLighting, HLPR_createWebGlBuffer, GL_handleMouseUp
 */

function BASE_PICK_customInitGL(canvas) {
    window.onresize = function () {
        updateGLCanvasSize(BRAIN_CANVAS_ID);
        LEG_updateLegendVerticesBuffers();
    };
    initGL(canvas);
    canvas.redrawFunctionRef = drawScene;
    drawingMode = gl.TRIANGLES;
}


function BASE_PICK_initShaders() {
    createAndUseShader("shader-fs", "shader-vs");
    SHADING_Context.surface_pick_init(GL_shaderProgram);

    gl.uniform1i(GL_shaderProgram.useVertexColors, true);
}


function BASE_PICK_webGLStart(urlVerticesPickList, urlTrianglesPickList, urlNormalsPickList, urlVerticesDisplayList,
                              urlTrianglesDisplayList, urlNormalsDisplayList, brain_center, callback, boundaryURL) {
    BRAIN_CENTER = $.parseJSON(brain_center);
    const canvas = document.getElementById(BRAIN_CANVAS_ID);

    BASE_PICK_customInitGL(canvas);
    BASE_PICK_initShaders();

    displayMessage("Start loading picking data!", "infoMessage");
    downloadBrainGeometry(urlVerticesPickList, urlTrianglesPickList, urlNormalsPickList,
        function (pickingBrain) {
            displayMessage("Finish loading picking data!", "infoMessage");
            BASE_PICK_brainPickingBuffers = drawingBrainUploadGeometryBuffers(pickingBrain);
            for (let i = 0; i < pickingBrain.vertices.length; i++) {
                verticesPoints.push(pickingBrain.vertices[i]);
            }
            __createPickingColorBuffers(pickingBrain.vertices, pickingBrain.indexes);
            executeCallback(callback);
            drawScene();
        }
    );

    displayMessage("Start loading surface data!", "infoMessage");
    downloadBrainGeometry(urlVerticesDisplayList, urlTrianglesDisplayList, urlNormalsDisplayList,
        function (drawingBrain) {
            displayMessage("Finished loading surface data!", "infoMessage");
            // Finished downloading buffer data. Initialize BASE_PICK_brainDisplayBuffers
            BASE_PICK_brainDisplayBuffers = drawingBrainUploadGeometryBuffers(drawingBrain);
            drawingBrainUploadDefaultColorBuffer(drawingBrain.vertices, BASE_PICK_brainDisplayBuffers);
            executeCallback(callback);
            drawScene();
            // keep the vertices. Their counts are used by BASE_PICK_updateBrainColors
            drawingBrainVertices = drawingBrain.vertices;
        }
    );

    initBrainNavigatorBuffers();
    createStimulusPinBuffers();

    ColSch_initColorSchemeComponent();
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    // Enable keyboard and mouse interaction
    canvas.onkeydown = customKeyDown;
    canvas.onkeyup = customKeyUp;
    canvas.onmousedown = customMouseDown;
    canvas.oncontextmenu = function () {
        return false;
    };
    $(document).on('mouseup', customMouseUp);
    $(document).on('mousemove', customMouseMove);

    $(canvas).mousewheel(function (event, delta) {
        BASE_PICK_handleMouseWeel(delta);
        return false; // prevent default
    });
    // Needed for when drawing the legend.
    isOneToOneMapping = true;

    drawScene();

    BASE_PICK_regionBoundariesController = new RB_RegionBoundariesController(boundaryURL);
}


/**
 * Redraw from buffers.
 */
function drawScene() {

    let brainBuffers;

    if (BASE_PICK_doPick) {
        brainBuffers = BASE_PICK_brainPickingBuffers;
    } else {
        brainBuffers = BASE_PICK_brainDisplayBuffers;
    }

    if (brainBuffers.length === 0) {
        displayMessage("The load operation for the surface data is not completed yet!", "infoMessage");
        return;
    }

    if (BASE_PICK_doPick) {
        gl.disable(gl.BLEND);
        gl.disable(gl.DITHER);
        basicSetLighting(pickingLightSettings);
    } else {
        gl.enable(gl.BLEND);
        gl.enable(gl.DITHER);
        basicSetLighting();
    }
    const theme = ColSchGetTheme().surfaceViewer;
    gl.clearColor(theme.backgroundColor[0], theme.backgroundColor[1], theme.backgroundColor[2], theme.backgroundColor[3]);
    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    // View angle is 45, we want to see object from 0.1 up to 800 distance from viewer
    const aspect = gl.viewportWidth / gl.viewportHeight;
    perspective(45, aspect, near, 800.0);

    mvPushMatrix();
    mvRotate(180, [0, 0, 1]);

    const col = ColSchInfo();
    const activityRange = ColSchGetBounds();
    SHADING_Context.colorscheme_set_uniforms(GL_shaderProgram, activityRange.min, activityRange.max,
        activityRange.bins, activityRange.centralHoleDiameter);
    gl.uniform1f(GL_shaderProgram.colorSchemeUniform, col.tex_v);

    gl.uniform1i(GL_shaderProgram.useActivity, !BASE_PICK_doPick);

    drawBuffers(drawingMode, brainBuffers);

    BASE_PICK_regionBoundariesController.drawRegionBoundaries(drawingMode, BASE_PICK_brainDisplayBuffers);

    gl.uniform1i(GL_shaderProgram.useActivity, false);

    if (!BASE_PICK_isMovieMode) { // todo does this navigator ever render? should it?
        mvPushMatrix();
        mvTranslate([navigatorX, navigatorY, navigatorZ]);
        mvRotate(navigatorXrot, [1, 0, 0]);
        mvRotate(navigatorYrot, [0, 1, 0]);
        drawBuffers(gl.TRIANGLES, [BASE_PICK_navigatorBuffers]);
        mvPopMatrix();
    }

    for (let key in surfaceFocalPoints) {
        const focalPointPosition = surfaceFocalPoints[key]['position'];
        mvPushMatrix();
        mvTranslate(focalPointPosition);
        mvRotate(surfaceFocalPoints[key]['xRotation'], [1, 0, 0]);
        mvRotate(surfaceFocalPoints[key]['yRotation'], [0, 1, 0]);
        drawBuffers(gl.TRIANGLES, [BASE_PICK_pinBuffers]);
        mvPopMatrix();
    }
    mvPopMatrix();
    gl.uniform1i(GL_shaderProgram.useActivity, true);

    if (BASE_PICK_brainDisplayBuffers.length !== 0 && LEG_legendBuffers.length) {
        // wait for the data to be loaded, then draw the legend
        mvPushMatrix();
        loadIdentity();
        drawBuffers(gl.TRIANGLES, [LEG_legendBuffers]);
        mvPopMatrix();
    }
}


function drawBuffers(drawMode, buffersSets) {
    setMatrixUniforms();
    for (let i = 0; i < buffersSets.length; i++) {
        SHADING_Context.surface_pick_draw(GL_shaderProgram,
            buffersSets[i][0], buffersSets[i][1],
            buffersSets[i][4], buffersSets[i][3],
            buffersSets[i][2], drawMode
        );
    }
}

/**
 * @param callback a string which should represents a valid js code.
 */
function executeCallback(callback) {
    if (callback == null || callback.trim().length === 0) {
        return;
    }
    if (BASE_PICK_brainPickingBuffers.length !== 0 && BASE_PICK_brainDisplayBuffers.length !== 0) {
        eval(callback);
    }
}

function __createPickingColorBuffers(pickingBrainVertices, pickingBrainIndexes) {
    let thisBufferColors;
    let total_picking_triangles_number = 0;

    for (let i = 0; i < pickingBrainVertices.length; i++) {
        const fakeActivityBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, fakeActivityBuffer);
        thisBufferColors = new Float32Array(pickingBrainVertices[i].length);
        gl.bufferData(gl.ARRAY_BUFFER, thisBufferColors, gl.STATIC_DRAW);

        picking_triangles_number.push(pickingBrainIndexes[i].length);
        total_picking_triangles_number = total_picking_triangles_number + pickingBrainIndexes[i].length;

        BASE_PICK_brainPickingBuffers[i][3] = fakeActivityBuffer;
    }

    GL_initColorPickFrameBuffer();
    //The total number of required colors will be equal to the total triangle number
    GL_initColorPickingData(total_picking_triangles_number);

    //Create the color buffer array for the picking part where each triangles has a new color
    let pointsSoFar = 0;

    for (let j = 0; j < picking_triangles_number.length; j++) {
        //For each set of triangles(different file) create a new color buffers
        const newColorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, newColorBuffer);
        thisBufferColors = new Float32Array(picking_triangles_number[j] * 4);
        //Go trough all the triangles from this set and set the same color for 3 adjacend vertices representing a triangle
        for (let idx = 0; idx < picking_triangles_number[j]; idx++) {
            const color = GL_colorPickerInitColors[(pointsSoFar + idx - (idx + pointsSoFar) % 3) / 3];
            thisBufferColors[4 * idx] = color[0];
            thisBufferColors[4 * idx + 1] = color[1];
            thisBufferColors[4 * idx + 2] = color[2];
            thisBufferColors[4 * idx + 3] = color[3];
        }
        //Since the colorPickingArray is not split we need to keep track of the absolute index of the triangles
        //considering all the files that were processed before, this pointsSoFar keeps track of this
        pointsSoFar += picking_triangles_number[j];
        gl.bufferData(gl.ARRAY_BUFFER, thisBufferColors, gl.STATIC_DRAW);

        BASE_PICK_brainPickingBuffers[j][4] = newColorBuffer;
    }
}

/**
 * Buffers the default gray surface color to the GPU
 * And updates BASE_PICK_brainDisplayBuffers[i] at indices 3 and 4
 */
function BASE_PICK_buffer_default_color() {
    drawingBrainUploadDefaultColorBuffer(drawingBrainVertices, BASE_PICK_brainDisplayBuffers);
}

/**
 * Update the buffers for coloring the displayed brain, with parameter "data"
 */
function BASE_PICK_updateBrainColors(data) {

    if (BASE_PICK_brainDisplayBuffers.length != data.length) {
        displayMessage("Could not draw the gradient colors. Try again in a few seconds, perhaps not everything is " +
            "fully loaded from the server.", "warningMessage");
        return;
    }

    for (let i = 0; i < data.length; i++) {
        const activityBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, activityBuffer);
        const activity = new Float32Array(data[i]);
        gl.bufferData(gl.ARRAY_BUFFER, activity, gl.STATIC_DRAW);
        activityBuffer.numItems = data[i].length;
        BASE_PICK_brainDisplayBuffers[i][3] = activityBuffer;
    }

    drawScene();
}

///////////////////////////////////////~~~~~~~~START MOUSE RELATED CODE~~~~~~~~~~~//////////////////////////////////

/**
 * When mouse is released check to see if the difference between when mouse was pressed
 * and mouse was released is higher than a threshold(now set to 5). If so the user either
 * rotated brain (when higher) or selected vertex (when lower).
 */
function customMouseUp(event) {
    GL_handleMouseUp(event);

    const canvasOffset = $("#" + BRAIN_CANVAS_ID).offset();
    const final_x_val = GL_lastMouseX + document.body.scrollLeft + document.documentElement.scrollLeft - Math.floor(canvasOffset.left);
    const final_y_val = GL_lastMouseY + document.body.scrollTop + document.documentElement.scrollTop - Math.floor(canvasOffset.top) + 1;
    const delta_x = Math.abs(final_x_val - GL_mouseXRelToCanvas);
    const delta_y = Math.abs(final_y_val - GL_mouseYRelToCanvas);

    if (delta_x <= 5 && delta_y <= 5) {
        BASE_PICK_doVerticePick();
    }

    if (!BASE_PICK_isMovieMode) {
        drawScene();
    }
}

function customMouseDown(event) {
    GL_handleMouseDown(event, $("#" + BRAIN_CANVAS_ID));
}

function customKeyDown(event) {
    GL_handleKeyDown(event);
    drawScene();
}

function customKeyUp(event) {
    GL_handleKeyUp(event);
    drawScene();
}

/**
 * Also redraw the scene on mouse move to give a more 'fluent' look
 */
function customMouseMove(event) {
    GL_handleMouseMove(event);
    if (!BASE_PICK_isMovieMode) {
        drawScene();
    }
}

function BASE_PICK_handleMouseWeel(event) {
    GL_handleMouseWeel(event);
    if (!BASE_PICK_isMovieMode) {
        drawScene();
    }
}

/**
 * Returns the TRIANGLE_pickedIndex of the "nearest" picked vertex index.
 * This should be 'fail-safe in case the clicked color is neither (0,0,0) nor in the picking dictionary. In this case just
 * start looking at adjacent pixels in pseudo-circular order until a proper one is found.
 */
function _BASE_PICK_find_picked_vertex() {
    const orig_x = GL_mouseXRelToCanvas;
    const orig_y = GL_mouseYRelToCanvas;
    const directions = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
    for (let inc = 0; inc <= 20; inc++) {
        for (let di = 0; di < directions.length; di++) {
            const dir = directions[di];
            GL_mouseXRelToCanvas = orig_x + inc * dir[0];
            GL_mouseYRelToCanvas = orig_y + inc * dir[1];
            TRIANGLE_pickedIndex = GL_getPickedIndex();
            if (TRIANGLE_pickedIndex != GL_NOTFOUND) {
                console.debug('found at delta ' + inc + " " + TRIANGLE_pickedIndex);
                return TRIANGLE_pickedIndex;
            }
        }
    }
    return TRIANGLE_pickedIndex;
}

/**
 * Draws the brain in picking mode, then finds the picked vertex. Also centers the navigator in that vertex.
 */
function BASE_PICK_doVerticePick() {
    if (BASE_PICK_brainPickingBuffers.length === 0 || BASE_PICK_brainDisplayBuffers.length === 0) {
        displayMessage("The load operation for the surface data is not completed yet!", "infoMessage");
        return;
    }
    //Drawing will be done to back buffer and all 'eye candy' is disabled to get pure color
    gl.bindFramebuffer(gl.FRAMEBUFFER, GL_colorPickerBuffer);

    //Draw the brain in picking mode: swap normal display colors with picking colors and draw the 'colored' version to a back buffer
    BASE_PICK_doPick = true;
    drawScene();
    TRIANGLE_pickedIndex = GL_getPickedIndex();

    if (TRIANGLE_pickedIndex != GL_BACKGROUND) {
        TRIANGLE_pickedIndex = _BASE_PICK_find_picked_vertex();
        BASE_PICK_moveBrainNavigator(false);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    BASE_PICK_doPick = false;
}

///////////////////////////////////////~~~~~~~~END MOUSE RELATED CODE~~~~~~~~~~~//////////////////////////////////

/////////////////////////////////////////~~~~~~~~Start Focal Points RELATED CODE~~~~~~~~~~~//////////////////////////////////
/*
 * Remove the focal point given by triangle index.
 */
function BASE_PICK_removeFocalPoint(triangleIndex) {
    delete surfaceFocalPoints[triangleIndex];
    drawScene();
}


function BASE_PICK_clearFocalPoints() {
    surfaceFocalPoints = {};
    drawScene();
}

/*
 * Store all required information in order to draw this focal point.
 */
function BASE_PICK_addFocalPoint(triangleIndex) {

    const x_rot = (360 - Math.atan2(navigatorY - BRAIN_CENTER[1], navigatorZ - BRAIN_CENTER[2]) * 180 / Math.PI) % 360;
    const y_rot = Math.atan2(navigatorX - BRAIN_CENTER[0], Math.sqrt((navigatorZ - BRAIN_CENTER[2]) * (navigatorZ - BRAIN_CENTER[2]) + (navigatorY - BRAIN_CENTER[1]) * (navigatorY - BRAIN_CENTER[1]))) * 180 / Math.PI;
    surfaceFocalPoints[triangleIndex] = {
        'xRotation': x_rot,
        'yRotation': y_rot,
        'position': [navigatorX, navigatorY, navigatorZ]
    };
    drawScene();
}

function createStimulusPinBuffers() {
    BASE_PICK_pinBuffers[0] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, BASE_PICK_pinBuffers[0]);
    const vertices = [-1, 0.5, 5,
        -0.5, 1, 5,
        0.5, 1, 5,
        1, 0.5, 5,
        1, -0.5, 5,
        0.5, -1, 5,
        -0.5, -1, 5,
        -1, -0.5, 5,
        0, 0, 0,
        -0.2, -0.2, 5,
        -0.2, 0.2, 5,
        0.2, -0.2, 5,
        0.2, 0.2, 5,
        -0.2, -0.2, 12,
        -0.2, 0.2, 12,
        0.2, -0.2, 12,
        0.2, 0.2, 12];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    BASE_PICK_pinBuffers[0].itemSize = 3;
    BASE_PICK_pinBuffers[0].numItems = 17;

    BASE_PICK_pinBuffers[1] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, BASE_PICK_pinBuffers[1]);
    const vertexNormals = [
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0
    ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertexNormals), gl.STATIC_DRAW);
    BASE_PICK_pinBuffers[1].itemSize = 3;
    BASE_PICK_pinBuffers[1].numItems = 17;

    BASE_PICK_pinBuffers[2] = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, BASE_PICK_pinBuffers[2]);
    const cubeVertexIndices = [0, 1, 5, 1, 2, 5,
        2, 3, 5, 3, 4, 5,
        6, 7, 5, 0, 7, 5,
        0, 1, 8, 1, 2, 8,
        2, 3, 8, 3, 4, 8,
        4, 5, 8, 5, 6, 8,
        6, 7, 7, 9, 10, 11,
        10, 11, 12, 13, 14, 15,
        14, 15, 16,
        9, 10, 13, 10, 13, 14,
        10, 12, 14, 12, 14, 16,
        11, 12, 16, 11, 15, 16,
        9, 11, 13, 11, 13, 15
    ];
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(cubeVertexIndices), gl.STATIC_DRAW);
    BASE_PICK_pinBuffers[2].itemSize = 1;
    BASE_PICK_pinBuffers[2].numItems = 75;

    let same_color = [];
    for (let i = 0; i < BASE_PICK_pinBuffers[0].numItems * 4; i++) {
        same_color = same_color.concat(1.0, 0.6, 0.0, 1.0);
    }
    BASE_PICK_pinBuffers[4] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, BASE_PICK_pinBuffers[4]);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(same_color), gl.STATIC_DRAW);
    // fake activity buffer
    BASE_PICK_pinBuffers[3] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, BASE_PICK_pinBuffers[3]);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(BASE_PICK_pinBuffers[0].numItems), gl.STATIC_DRAW);
}


/////////////////////////////////////////~~~~~~~~END FP RELATED CODE~~~~~~~~~~~//////////////////////////////////

////////////////////////////////////~~~~~~~~START BRAIN NAVIGATOR RELATED CODE~~~~~~~~~~~///////////////////////////

function initBrainNavigatorBuffers() {
    BASE_PICK_navigatorBuffers[0] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, BASE_PICK_navigatorBuffers[0]);
    const vertices = [-0.2, -0.2, 0,
        -0.2, 0.2, 0,
        0.2, -0.2, 0,
        0.2, 0.2, 0,
        -0.2, -0.2, 12,
        -0.2, 0.2, 12,
        0.2, -0.2, 12,
        0.2, 0.2, 12,
        -1, 0.5, 12,
        -0.5, 1, 12,
        0.5, 1, 12,
        1, 0.5, 12,
        1, -0.5, 12,
        0.5, -1, 12,
        -0.5, -1, 12,
        -1, -0.5, 12];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    BASE_PICK_navigatorBuffers[0].itemSize = 3;
    BASE_PICK_navigatorBuffers[0].numItems = 16;

    BASE_PICK_navigatorBuffers[1] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, BASE_PICK_navigatorBuffers[1]);
    const vertexNormals = [
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0
    ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertexNormals), gl.STATIC_DRAW);
    BASE_PICK_navigatorBuffers[1].itemSize = 3;
    BASE_PICK_navigatorBuffers[1].numItems = 12;

    BASE_PICK_navigatorBuffers[2] = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, BASE_PICK_navigatorBuffers[2]);
    const cubeVertexIndices = [0, 1, 2, 1, 2, 3,    // z plane - far
        4, 5, 6, 5, 6, 7,    // z plane - near - small
        8, 9, 13, 9, 10, 13,  // z plane - near - big
        10, 11, 13, 11, 12, 13,
        14, 15, 13, 8, 15, 13,
        0, 1, 4, 1, 4, 5,      // 'walls'
        1, 3, 5, 3, 5, 7,
        2, 3, 7, 2, 6, 7,
        0, 2, 4, 2, 4, 6
    ];
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(cubeVertexIndices), gl.STATIC_DRAW);
    BASE_PICK_navigatorBuffers[2].itemSize = 1;
    BASE_PICK_navigatorBuffers[2].numItems = 54;
    let same_color = [];
    for (let i = 0; i < BASE_PICK_navigatorBuffers[0].numItems * 4; i++) {
        same_color = same_color.concat(0.0, 0.0, 1.0, 1.0);
    }
    BASE_PICK_navigatorBuffers[4] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, BASE_PICK_navigatorBuffers[4]);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(same_color), gl.STATIC_DRAW);
    // fake activity
    BASE_PICK_navigatorBuffers[3] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, BASE_PICK_navigatorBuffers[3]);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(BASE_PICK_navigatorBuffers[0].numItems), gl.STATIC_DRAW);
}

/**
 * Moves the brain navigator to a certain position.
 */
function BASE_PICK_moveBrainNavigator(shouldRedrawScene) {
    if (TRIANGLE_pickedIndex != GL_NOTFOUND && TRIANGLE_pickedIndex != GL_BACKGROUND) {
        // If we managed to find an index find the triangle that corresponds to it from the list of draw triangles
        // Then take the first vertex as new navigator coordinate. Potentially this could be replaced with some other
        // metric, like the center of the triangle, or the vertex closest the the other two etc.

        let pickedIndex = TRIANGLE_pickedIndex;
        for (let j = 0; j < picking_triangles_number.length; j++) {
            if (pickedIndex < picking_triangles_number[j] / 3) {
                navigatorX = parseFloat(verticesPoints[j][9 * pickedIndex]);
                navigatorY = parseFloat(verticesPoints[j][9 * pickedIndex + 1]);
                navigatorZ = parseFloat(verticesPoints[j][9 * pickedIndex + 2]);
                break;
            } else {
                pickedIndex = pickedIndex - picking_triangles_number[j] / 3;
            }
        }
        VERTEX_pickedIndex = pickedIndex;
    }
    navigatorXrot = (360 - Math.atan2(navigatorY - BRAIN_CENTER[1], navigatorZ - BRAIN_CENTER[2]) * 180 / Math.PI) % 360;
    navigatorYrot = Math.atan2(navigatorX - BRAIN_CENTER[0], Math.sqrt((navigatorZ - BRAIN_CENTER[2]) * (navigatorZ - BRAIN_CENTER[2]) + (navigatorY - BRAIN_CENTER[1]) * (navigatorY - BRAIN_CENTER[1]))) * 180 / Math.PI;

    if (shouldRedrawScene) {
        drawScene();
    }
}

function BASE_PICK_moveNavigatorToCenter() {
    navigatorX = BRAIN_CENTER[0];
    navigatorY = BRAIN_CENTER[1];
    navigatorZ = BRAIN_CENTER[2];
    TRIANGLE_pickedIndex = GL_NOTFOUND;
}

////////////////////////////////////~~~~~~~~END BRAIN NAVIGATOR RELATED CODE~~~~~~~~~~~/////////////////////////////


/**
 * Init the legend labels, from minValue to maxValue
 *
 * @param maxValue Maximum value for the gradient end
 * @param minValue Minimum value for the gradient start
 */
function BASE_PICK_initLegendInfo(maxValue, minValue) {
    const brainLegendDiv = document.getElementById('brainLegendDiv');
    ColSch_updateLegendLabels(brainLegendDiv, minValue, maxValue, "100%");
}

