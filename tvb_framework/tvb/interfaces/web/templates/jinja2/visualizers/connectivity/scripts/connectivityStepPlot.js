/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need to download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

var CONNECTIVITY_SPACE_TIME_CANVAS_ID = "GLcanvas_SPACETIME";

// The Space Time plot is drawn with three.js (window.THREE, loaded as an ES module in main_connectivity.html).
var spaceTimeRenderer = null;       // three.js renderer on the Space Time canvas, created once
var spaceTimeScene = null;          // three.js scene holding the matrices
var spaceTimeCamera = null;         // three.js camera
var spaceTimeGeometries = null;     // Geometries shared by all matrices: rectangle, lines between cells, outline
var spaceTimePlots = [];            // One entry per matrix: {group, plane, grid, outline}
var SPACE_TIME_MIN_GRID_CELL = 3;   // Smallest cell size (in plot units, about css px) for which the lines between cells are drawn

var plotSize = 250;                    // Default plot size of 250 pixels
var defaultZ = -2.0;                // Default plot Z position of -2.0
var nrOfSteps = 6;                    // Number of space-time plots we will draw in scene

var plotTranslations = [];            // Keep track of the translation for each plot.
var plotRotations = [];                // Keep track of the rotations for each plot
var zoomedInMatrix = -1;            // The matrix witch is currently zoomed in on
var clickedMatrix = -1;
var backupTranslations = [];
var backupRotations = [];
var animationStarted = false;
var alphaValueSpaceTime = 1.0;                // original alpha value for default plot
var backupAlphaValue = alphaValueSpaceTime;    // backup used in animations
var minSelectedDelayValue = -1;
var maxSelectedDelayValue = -1;
var animationTimeout = 33; // 30Hz
var animationGranularity = 20;


function customMouseDown_SpaceTime(event) {
    if (!animationStarted) {
        if (clickedMatrix >= 0) {
            doZoomOutAnimation();
        } else {
            clickedMatrix = _pickSpaceTimePlot(event);
            if (clickedMatrix >= 0) {
                doZoomInAnimation();
            }
        }
    }
}

/*
 * Return the index of the matrix under the mouse, or -1 when there is none.
 */
function _pickSpaceTimePlot(event) {
    var rect = spaceTimeRenderer.domElement.getBoundingClientRect();
    // Mouse position in normalized device coordinates, from -1 to 1 on both axes
    var mouse = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1,
                                  -((event.clientY - rect.top) / rect.height) * 2 + 1);
    var raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, spaceTimeCamera);
    var planes = spaceTimePlots.map(function (plot) { return plot.plane; });
    spaceTimeScene.updateMatrixWorld();
    var hits = raycaster.intersectObjects(planes, false);
    return hits.length ? planes.indexOf(hits[0].object) : -1;
}


function connectivitySpaceTime_startGL() {
    conectivitySpaceTime_initCanvas();
    updateSpaceTimeHeader();
}

/*
 * Create the geometries shared by all the matrices of the space-time plot.
 * Each matrix is a single rectangle with a texture holding one texel per cell, so the number of
 * vertices does not depend on the number of regions.
 */
function createConnectivityMatrix() {
    var nrElems = GVAR_interestAreaVariables[GVAR_selectedAreaType].values.length;
    // starting 'x' and 'y' axis values for the plot in order to center around (0, 0)
    var startX = - plotSize / 2;
    var startY = - plotSize / 2;
    // The size of a matrix element
    var elementSize = plotSize / nrElems;

    if (spaceTimeGeometries) {
        for (var key in spaceTimeGeometries) {
            if (spaceTimeGeometries[key]) {
                spaceTimeGeometries[key].dispose();
            }
        }
    }
    spaceTimeGeometries = {
        // Centered on (0, 0), with texture coordinates from (0, 0) at (startX, startY) to (1, 1) at the opposite corner
        plane: new THREE.PlaneGeometry(plotSize, plotSize),
        grid: elementSize >= SPACE_TIME_MIN_GRID_CELL ? createGridLines(startX, startY, elementSize, nrElems) : null,
        outline: _createLinesGeometry([
            startX, startY, defaultZ,
            startX + plotSize, startY, defaultZ,
            startX + plotSize, startY + plotSize, defaultZ,
            startX, startY + plotSize, defaultZ
        ])
    };
}

/*
 * Compute the lines between the matrix cells: nrElems + 1 vertical and nrElems + 1 horizontal lines.
 */
function createGridLines(startX, startY, elementSize, nrElems) {
    var width = nrElems * elementSize;
    var lines = new Float32Array((nrElems + 1) * 4 * 3);
    for (var k = 0; k <= nrElems; k++) {
        var x = startX + k * elementSize;
        var y = startY + k * elementSize;
        lines.set([
            x, startY, defaultZ, x, startY + width, defaultZ,
            startX, y, defaultZ, startX + width, y, defaultZ
        ], k * 4 * 3);
    }
    return _createLinesGeometry(lines);
}

function _createLinesGeometry(positions) {
    var geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    return geometry;
}

/*
 * Create the three.js objects of one matrix, drawn with the given texture.
 */
function _createSpaceTimePlot(texture) {
    var theme = ColSchGetTheme().connectivityStepPlot;
    var plane = new THREE.Mesh(spaceTimeGeometries.plane,
                               new THREE.MeshBasicMaterial({map: texture, side: THREE.DoubleSide, transparent: true}));
    plane.position.z = defaultZ;
    var outline = new THREE.LineLoop(spaceTimeGeometries.outline, new THREE.LineBasicMaterial({transparent: true}));
    // Lines are drawn after the rectangles, so they stay visible on top of them
    outline.renderOrder = 2;
    // The matrix is turned upside down, as in the WebGL version (rotation of 180 degrees around z)
    var flipped = new THREE.Group();
    flipped.rotation.z = Math.PI;
    flipped.add(plane, outline);
    var grid = null;
    if (spaceTimeGeometries.grid) {
        grid = new THREE.LineSegments(spaceTimeGeometries.grid, new THREE.LineBasicMaterial({transparent: true}));
        grid.material.color.setRGB(theme.lineColor[0], theme.lineColor[1], theme.lineColor[2]);
        grid.renderOrder = 1;
        flipped.add(grid);
    }
    var group = new THREE.Group();
    group.add(flipped);
    spaceTimeScene.add(group);
    return {group: group, plane: plane, grid: grid, outline: outline};
}

/*
 * Remove all the matrices from the scene and free their textures and materials.
 */
function _disposeSpaceTimePlots() {
    for (var plot of spaceTimePlots) {
        spaceTimeScene.remove(plot.group);
        plot.plane.material.map.dispose();
        plot.plane.material.dispose();
        plot.outline.material.dispose();
        if (plot.grid) {
            plot.grid.material.dispose();
        }
    }
    spaceTimePlots = [];
}


/*
 * Generate a texture which represents the state of the weights for
 * a given 'interval' centered around a given tract value
 *
 * @param tractValue: the center of the interval
 * @param intervalLength: the length of the interval
 *
 * @returns: a texture with one texel per matrix cell, where for the connections that fall in the defined interval,
 *              a gradient color is assigned based on the weights strenght, and for the
 *              rest the no value color is used.
 */
function generateTexture(tractValue, intervalLength) {
    var theme = ColSchGetTheme().connectivityStepPlot;
    var matrixWeightsValues = GVAR_interestAreaVariables[1].values;
    var matrixTractsValues = GVAR_interestAreaVariables[2].values;
    var minWeightsValue = GVAR_interestAreaVariables[1].min_val;
    var maxWeightsValue = GVAR_interestAreaVariables[1].max_val;
    var nrElems = matrixWeightsValues.length;
    var texels = new Uint8Array(nrElems * nrElems * 4);
    var linearIndex = -1;

    for (var i = 0; i < nrElems; i++) {
        for (var j = 0; j < nrElems; j++) {
            linearIndex += 1;
            var delayValue = matrixTractsValues[i][nrElems - j - 1] / conductionSpeed;
            var weight = matrixWeightsValues[i][nrElems - j - 1];

            var isWithinInterval = (delayValue >= (tractValue - intervalLength / 2) &&
                                    delayValue <= (tractValue + intervalLength / 2));
            var color;

            if (isWithinInterval && weight != 0) {
                color = getGradientColor(weight, minWeightsValue, maxWeightsValue).slice(0, 3);
            }else{
                color = theme.noValueColor;
            }

            texels.set([color[0] * 255, color[1] * 255, color[2] * 255, 255], 4 * linearIndex);
        }
    }
    var texture = new THREE.DataTexture(texels, nrElems, nrElems, THREE.RGBAFormat, THREE.UnsignedByteType);
    // No filtering, so that each cell keeps its own color.
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.NearestFilter;
    texture.needsUpdate = true;
    return texture;
}


/**
 * Update the header with information about matrices.
 */
function updateSpaceTimeHeader() {

    function parseFloatDefault(valueUI, defaultValue){
        var resultValue = parseFloat(valueUI);
        if(isNaN(resultValue)){
            return defaultValue;
        }else{
            return resultValue;
        }
    }

    var fromDelaysInput = $('#fromDelaysValue');
    var toDelaysInput = $('#toDelaysValue');
    var conductionSpeedInput = $('#conductionSpeedValue');
    var currentMinDelay, currentMaxDelay;
    var uiConductionSpeed = parseFloatDefault(conductionSpeedInput.val(), conductionSpeed);

    // if conduction speed has changed adjust tract bounds
    if(uiConductionSpeed !== conductionSpeed){
        conductionSpeed = uiConductionSpeed;
        currentMinDelay = GVAR_interestAreaVariables[2].min_val / conductionSpeed;
        currentMaxDelay = GVAR_interestAreaVariables[2].max_val / conductionSpeed;
        _connectivitySpaceTimeUpdateLegend();

    } else {
        // read delay bounds specified by user in UI, in case invalid defaults are previous values
        currentMinDelay = parseFloatDefault(fromDelaysInput.val(), minSelectedDelayValue);
        currentMaxDelay = parseFloatDefault(toDelaysInput.val(), maxSelectedDelayValue);

        // ensure validity for tract bounds
        var maxAcceptedDelay = GVAR_interestAreaVariables[2].max_val / conductionSpeed;
        var minAcceptedDelay = GVAR_interestAreaVariables[2].min_val / conductionSpeed;

        if (currentMinDelay > currentMaxDelay) {
            var swapAux = currentMinDelay;
            currentMinDelay = currentMaxDelay;
            currentMaxDelay = swapAux;
        }
        if (currentMinDelay < minAcceptedDelay){
            currentMinDelay = minAcceptedDelay;
        }
        if (currentMaxDelay > maxAcceptedDelay || currentMaxDelay < 0) {
            currentMaxDelay = maxAcceptedDelay;
        }
    }

    //set globals
    minSelectedDelayValue = currentMinDelay;
    maxSelectedDelayValue = currentMaxDelay;

    //update ui
    fromDelaysInput.val(minSelectedDelayValue.toFixed(2));
    toDelaysInput.val(maxSelectedDelayValue.toFixed(2));
    conductionSpeedInput.val(conductionSpeed.toFixed(2));

    ConnStepPlotCreateMatrices();
    if (clickedMatrix >= 0) {
        doZoomOutAnimation();
    } else {
        drawSceneSpaceTime();
    }
}


function ConnStepPlotCreateMatrices() {
    _disposeSpaceTimePlots();
    var stepValue = (maxSelectedDelayValue - minSelectedDelayValue) / nrOfSteps;
    spaceTimePlots.push(_createSpaceTimePlot(generateTexture((maxSelectedDelayValue + minSelectedDelayValue) / 2, maxSelectedDelayValue - minSelectedDelayValue)));
    // One slice per step, counted by index so rounding cannot drop or add a slice.
    // When From equals To there is nothing to split and only the full matrix is shown.
    if (stepValue > 0) {
        for (var step = 0; step < nrOfSteps; step++) {
            var tractValue = minSelectedDelayValue + stepValue * (step + 0.5);
            spaceTimePlots.push(_createSpaceTimePlot(generateTexture(tractValue, stepValue)));
        }
    }
    var theme = ColSchGetTheme().connectivityStepPlot;
    spaceTimeRenderer.setClearColor(new THREE.Color().setRGB(theme.backgroundColor[0], theme.backgroundColor[1],
                                                             theme.backgroundColor[2]), theme.backgroundColor[3]);
    drawSceneSpaceTime();
}

/*
 * Initialize the space time connectivity plot.
 */
function conectivitySpaceTime_initCanvas() {
    var canvas = document.getElementById(CONNECTIVITY_SPACE_TIME_CANVAS_ID);
    if (!spaceTimeRenderer) {
        // Created once: the canvas keeps the same renderer (and WebGL context) when the tab is opened again.
        spaceTimeRenderer = new THREE.WebGLRenderer({canvas: canvas, antialias: true});
        // TVB colors are used as they are, without color space conversion, like in the other WebGL viewers.
        spaceTimeRenderer.outputColorSpace = THREE.LinearSRGBColorSpace;
        spaceTimeScene = new THREE.Scene();
        // View angle is 45, we want to see object from near up to 800 distance from viewer, placed 600 away.
        spaceTimeCamera = new THREE.PerspectiveCamera(45, 1, near, 800.0);
        spaceTimeCamera.position.set(0, 0, 600);
    }
    plotSize = parseInt(canvas.clientWidth / 3);    // Compute the size of one connectivity plot depending on the canvas width
    createConnectivityMatrix();
    canvas.onmousedown = customMouseDown_SpaceTime;

    plotTranslations = [];
    plotRotations = [];
    plotTranslations.push([-parseInt(canvas.clientWidth / 4), 0, 0]);    //The translation for the left-most full connectivity matrix
    plotRotations.push([90, [0, 1, 0]]);
    for (var i = 0; i < nrOfSteps; i++) {
        plotTranslations.push([-parseInt(canvas.clientWidth / 8) + parseInt(canvas.clientWidth / 2.2 / nrOfSteps) * i, 0.0, 0.0]); // Values tested that display nicely for 6 plots at least
        plotRotations.push([80 - i * nrOfSteps, [0, 1, 0]]);
    }

    if (minSelectedDelayValue < 0) {
        minSelectedDelayValue = GVAR_interestAreaVariables[2].min_val / conductionSpeed;
    }
    if (maxSelectedDelayValue < 0) {
        maxSelectedDelayValue = GVAR_interestAreaVariables[2].max_val / conductionSpeed;
    }

    clickedMatrix = -1;

    _connectivitySpaceTimeUpdateLegend();
}

function _connectivitySpaceTimeUpdateLegend(){
    $('#leg_min_tract').html(GVAR_interestAreaVariables[2].min_non_zero.toFixed(2));
    $('#leg_max_tract').html(GVAR_interestAreaVariables[2].max_val.toFixed(2));
    $('#leg_min_weights').html(GVAR_interestAreaVariables[1].min_non_zero.toFixed(2));
    $('#leg_max_weights').html(GVAR_interestAreaVariables[1].max_val.toFixed(2));
    $('#leg_conduction_speed').html(conductionSpeed.toFixed(2));
    $('#leg_min_delay').html((GVAR_interestAreaVariables[2].min_non_zero / conductionSpeed).toFixed(2));
    $('#leg_max_delay').html((GVAR_interestAreaVariables[2].max_val / conductionSpeed).toFixed(2));
}


function computeAnimationTranslations(startPoint, endPoint, startRotation, endRotation, startAlpha, endAlpha, animationSteps) {
    var translationAnimation = [];
    var rotationAnimation = [];
    var alphas = [];
    // y-axis component does not vary so we only handle x and z linearly
    var x_inc = (endPoint[0] - startPoint[0]) / animationSteps;
    var z_inc = (endPoint[2] - startPoint[2]) / animationSteps;
    // also we do rotations just along y-axis so angle of rotation is all we need to change
    var rot_inc = (endRotation[0] - startRotation[0]) / animationSteps;
    // for alpha alos vary linearly
    var alpha_inc = (endAlpha - startAlpha) / animationSteps;
    for (var i = 1; i <= animationSteps; i++) {
        translationAnimation.push([startPoint[0] + i * x_inc, startPoint[1], startPoint[2] + i * z_inc]);
        rotationAnimation.push([startRotation[0] + i * rot_inc, startRotation[1]]);
        alphas.push(startAlpha + alpha_inc * i);
    }
    return {'translations' : translationAnimation, 'rotations' : rotationAnimation, 'alphas' : alphas};
}


function animationStep(step, animationSteps, animations, zoomIn) {
    for (var j = 0; j < animations.length; j++) {
        plotTranslations[j] = animations[j]['translations'][step];
        plotRotations[j] = animations[j]['rotations'][step];
        alphaValueSpaceTime = animations[j]['alphas'][step];
    }
    drawSceneSpaceTime();
    if (step + 1 < animationSteps) {
        setTimeout(function() { animationStep(step + 1, animationSteps, animations, zoomIn); }, animationTimeout);
    } else {
        var matrixSelected = document.getElementById('selectedMatrixValue');
        if (zoomIn) {
            zoomedInMatrix = clickedMatrix;
            var stepValue = (maxSelectedDelayValue - minSelectedDelayValue) / nrOfSteps;
            if (zoomedInMatrix != 0) {
                var fromTractVal = (minSelectedDelayValue + stepValue * (zoomedInMatrix - 1)).toFixed(2);
                var toTractVal = (minSelectedDelayValue + stepValue * zoomedInMatrix).toFixed(2);
                matrixSelected.innerHTML = '[' + fromTractVal + '..' + toTractVal + ']';
            } else {
                matrixSelected.innerHTML = 'Full matrix';
            }
        } else {
            plotTranslations = backupTranslations;
            plotRotations = backupRotations;
            alphaValueSpaceTime = backupAlphaValue;
            clickedMatrix = -1;
            zoomedInMatrix = -1;
            matrixSelected.innerHTML = 'None';
        }
        drawSceneSpaceTime();
        animationStarted = false;
    }
}


function doZoomInAnimation() {
    animationStarted = true;
    var targetForwardPosition = [0.0, 0.0, 200];
    var targetForwardRotation = [360, [0, 1, 0]];
    backupTranslations = plotTranslations.slice(0);
    backupRotations  = plotRotations.slice(0);
    backupAlphaValue = alphaValueSpaceTime;
    var animations = [];
    for (var i = 0; i < plotTranslations.length; i++) {
        var targetTranslation, endRotation;
        if (i == clickedMatrix) {
            targetTranslation = targetForwardPosition;
            endRotation = targetForwardRotation;
        } else {
            targetTranslation = [plotTranslations[i][0], plotTranslations[i][1], -200];
            endRotation = plotRotations[i];
        }
        animations.push(computeAnimationTranslations(plotTranslations[i], targetTranslation,
                                                     plotRotations[i], endRotation,
                                                     alphaValueSpaceTime, 1, animationGranularity));
    }
    animationStep(0, animationGranularity, animations, true);
}


function doZoomOutAnimation() {
    animationStarted = true;
    var animations = [];
    for (var i = 0; i < plotTranslations.length; i++) {
        animations.push(computeAnimationTranslations(plotTranslations[i], backupTranslations[i],
                                                     plotRotations[i], backupRotations[i],
                                                     _alphaValue, backupAlphaValue, animationGranularity));
    }
    animationStep(0, animationGranularity, animations, false);
}


/*
 * Draw the entire space plot matrices.
 */
function drawSceneSpaceTime() {
    if (!spaceTimeRenderer) {
        return;
    }
    _syncSpaceTimeSize();
    var theme = ColSchGetTheme().connectivityStepPlot;
    var axis = new THREE.Vector3();
    for (var i = 0; i < spaceTimePlots.length; i++) {
        var plot = spaceTimePlots[i];
        // Translate and rotate to get a good view
        plot.group.position.fromArray(plotTranslations[i]);
        plot.group.setRotationFromAxisAngle(axis.fromArray(plotRotations[i][1]), THREE.MathUtils.degToRad(plotRotations[i][0]));
        var outlineColor = (i === clickedMatrix) ? theme.selectedOutlineColor : theme.outlineColor;
        plot.outline.material.color.setRGB(outlineColor[0], outlineColor[1], outlineColor[2]);
        for (var material of [plot.plane.material, plot.outline.material, plot.grid ? plot.grid.material : null]) {
            if (material) {
                material.opacity = alphaValueSpaceTime;
            }
        }
    }
    spaceTimeRenderer.render(spaceTimeScene, spaceTimeCamera);
}

/*
 * The canvas size is set by updateGLCanvasSize on resize; keep the renderer and the camera in sync with it.
 */
function _syncSpaceTimeSize() {
    var canvas = spaceTimeRenderer.domElement;
    var size = spaceTimeRenderer.getSize(new THREE.Vector2());
    if (size.x !== canvas.width || size.y !== canvas.height) {
        spaceTimeRenderer.setSize(canvas.width, canvas.height, false);
        spaceTimeCamera.aspect = canvas.width / canvas.height;
        spaceTimeCamera.updateProjectionMatrix();
    }
}