// TODO: add legend, labels on axes, color scheme support
var ctx = null;                                                      // the context for drawing on current canvas
var currentQuadrant, quadrants = [];
var minimumValue, maximumValue, data;                                // minimum and maximum for current data slice
var voxelSize, volumeOrigin;                                         // volumeOrigin is not used for now, as in 2D it
                                                                    // is irrelevant; if needed, use it _setQuadrant
var selectedEntity = [0, 0, 0];                                      // the selected voxel; [i, j, k]
var quadrantHeight, quadrantWidth;

var Quadrant = function (params) {                                  // this keeps all necessary data for drawing
    this.index = params.index || 0;                                  // in a quadrant
    this.axes = params.axes || {x: 0, y: 1};                         // axes represented in current quad; i=0, j=1, k=2
    this.entityWidth = params.entityWidth || 0;                      // width and height of one voxel in current quad
    this.entityHeight = params.entityHeight || 0;
    this.offsetX = params.offsetX || 0;                              // the offset of the drawing relative to the quad
    this.offsetY = params.offsetY || 0;
};

/**
 * Make all the necessary initialisations and draws the default view, with the center voxel selected
 * @param dataUrls  Urls containing data slices from server
 * @param minValue  The minimum value for all the slices
 * @param maxValue  The maximum value for all the slices
 * @param volOrigin The origin of the rendering; irrelevant in 2D, for now
 * @param sizeOfVoxel   How the voxel is sized on each axis; [xScale, yScale, zScale]
 * @param voxelUnit The unit used for this rendering ("mm", "cm" etc)
 */
function startVisualiser(dataUrls, minValue, maxValue, volOrigin, sizeOfVoxel, voxelUnit) {
    var canvas = document.getElementById("volumetric-ts-canvas");
    if (!canvas.getContext) {
        displayMessage('You need a browser with canvas capabilities, to see this demo fully!', "errorMessage");
        return
    }

    volumeOrigin = $.parseJSON(volOrigin)[0];
    voxelSize    = $.parseJSON(sizeOfVoxel);

    canvas.width  = $(canvas).parent().width();                      // fill the screen on width
    canvas.height = canvas.width / 3 + 100;                          // three quadrants + some space for labeling
    quadrantHeight = quadrantWidth = canvas.width / 3;               // quadrants are squares

    ctx = canvas.getContext("2d");

    dataUrls = $.parseJSON(dataUrls);
    data = HLPR_readJSONfromFile(dataUrls[0]);
    data = data[0];                                                  // just the first slice for now

    _rotateData();                                                   // rotate Z axis
    minimumValue =  9999;                                            // compute the minimum on this slice
    maximumValue = -9999;
    for (var i = 0; i < data.length; ++i)
        for (var j = 0; j < data[0].length; ++j)
            for (var k = 0; k < data[0][0].length; ++k)
                if (data[i][j][k] > maximumValue)
                    maximumValue = data[i][j][k];
                else if (data[i][j][k] < minimumValue)
                    minimumValue = data[i][j][k];

    _setupQuadrants();

    selectedEntity[0] = Math.floor(data.length / 2);                 // set the center entity as the selected one
    selectedEntity[1] = Math.floor(data[0].length / 2);
    selectedEntity[2] = Math.floor(data[0][0].length / 2);

    drawScene();
}

// ==================================== DRAWING FUNCTIONS START =============================================

/**
 * Draws the current view depending on the selected entity
 */
// TODO: since only two dimensions change at every time, redraw just those quadrants
function drawScene() {
    _setCtxOnQuadrant(0);
    ctx.fillStyle = getGradientColorString(minimumValue, minimumValue, maximumValue);
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    for (var j = 0; j < data[0].length; ++j)
        for (var i = 0; i < data.length; ++i)
            drawVoxel(i, j, data[i][j][selectedEntity[2]]);

    _setCtxOnQuadrant(1);
    for (var k = 0; k < data[0][0].length; ++k)
        for (var jj = 0; jj < data[0].length; ++jj)
            drawVoxel(k, jj, data[selectedEntity[0]][jj][k]);

    _setCtxOnQuadrant(2);
    for (var kk = 0; kk < data[0][0].length; ++kk)
        for (var ii = 0; ii < data.length; ++ii)
            drawVoxel(kk, ii, data[ii][selectedEntity[1]][kk]);
    drawNavigator();
}

/**
 * Draws the voxel set at (line, col) in the current quadrant, and colors it according to its value
 */
function drawVoxel(line, col, value) {
    ctx.fillStyle = getGradientColorString(value, minimumValue, maximumValue);
    // col increases horizontally and line vertically, so col represents the X drawing axis, and line the Y
    ctx.fillRect(col * currentQuadrant.entityWidth, line * currentQuadrant.entityHeight,
                 currentQuadrant.entityWidth, currentQuadrant.entityHeight);
}

/**
 * Draws the cross-hair on each quadrant, on the <code>selectedEntity</code>
 */
function drawNavigator() {
    ctx.save();
    ctx.beginPath();

    for (var quadIdx = 0; quadIdx < 3; ++quadIdx) {
        _setCtxOnQuadrant(quadIdx);
        drawCrossHair(selectedEntity[currentQuadrant.axes.x] * currentQuadrant.entityWidth + currentQuadrant.entityWidth / 2,
                      selectedEntity[currentQuadrant.axes.y] * currentQuadrant.entityHeight + currentQuadrant.entityHeight / 2);
    }
    ctx.strokeStyle = "blue";
    ctx.stroke();
    ctx.restore();
}

/**
 * Draws a 20px X 20px cross hair on the <code>currentQuadrant</code>, at the specified x and y
 */
function drawCrossHair(x, y) {
    ctx.moveTo(Math.max(x - 20, 0), y);                              // the horizontal line
    ctx.lineTo(Math.min(x + 20, quadrantWidth), y);
    ctx.moveTo(x, Math.max(y - 20, 0));                              // the vertical line
    ctx.lineTo(x, Math.min(y + 20, quadrantHeight));
}

// ==================================== DRAWING FUNCTIONS  END  =============================================

// ==================================== PRIVATE FUNCTIONS START =============================================

/**
 * Sets the <code>currentQuadrant</code> and applies transformations on context depending on that
 *
 * @param quadIdx Specifies which of <code>quadrants</code> is selected
 * @private
 */
/* TODO: make it use volumeOrigin; could be like this:
 * <code>ctx.setTransform(1, 0, 0, 1, volumeOrigin[currentQuadrant.axes.x], volumeOrigin[currentQuadrant.axes.y])</code>
 *       if implemented, also change the picking to take it into account
 */
function _setCtxOnQuadrant(quadIdx) {
    currentQuadrant = quadrants[quadIdx];
    ctx.setTransform(1, 0, 0, 1, 0, 0);                              // reset the transformation
    ctx.translate(quadIdx * quadrantWidth + currentQuadrant.offsetX, currentQuadrant.offsetY)
}

/**
 * Rotates the K axis on the data to get a nice, upright view of the brain
 */
function _rotateData() {
    for (var i = 0; i < data.length; ++i)
        for (var j = 0; j < data[0].length; ++j)
            data[i][j].reverse()
}

/**
 * Returns the number of elements on the given axis
 * @param axis The axis whose length is returned; i=0, j=1, k=2
 * @returns {*}
 * @private
 */
function _getDataSize(axis) {
    switch (axis) {
        case 0:     return data.length;
        case 1:     return data[0].length;
        case 2:     return data[0][0].length
    }
}

/**
 * Computes the actual dimension of one entity from the specified axes
 * @param xAxis The axis to be represented on X (i=0, j=1, k=2)
 * @param yAxis The axis to be represented on Y (i=0, j=1, k=2)
 * @returns {{width: number, height: number}} Entity width and height
 */
function _getEntityDimensions(xAxis, yAxis) {
    var scaleOnWidth  = quadrantWidth  / (_getDataSize(xAxis) * voxelSize[xAxis]);
    var scaleOnHeight = quadrantHeight / (_getDataSize(yAxis) * voxelSize[yAxis]);
    var scale = Math.min(scaleOnHeight, scaleOnWidth);
    return {width: voxelSize[xAxis] * scale, height: voxelSize[yAxis] * scale}
}

/**
 * Initializes the <code>quadrants</code> with some default axes and sets their properties
 */
function _setupQuadrants() {
    quadrants.push(new Quadrant({ index: 0, axes: {x: 1, y: 0} }));
    quadrants.push(new Quadrant({ index: 1, axes: {x: 1, y: 2} }));
    quadrants.push(new Quadrant({ index: 2, axes: {x: 0, y: 2} }));

    for (var quadIdx = 0; quadIdx < quadrants.length; ++quadIdx) {
        var entityDimensions = _getEntityDimensions(quadrants[quadIdx].axes.x, quadrants[quadIdx].axes.y);
        quadrants[quadIdx].entityHeight = entityDimensions.height;
        quadrants[quadIdx].entityWidth  = entityDimensions.width;
        var drawingHeight = _getDataSize(quadrants[quadIdx].axes.y) * quadrants[quadIdx].entityHeight;
        var drawingWidth  = _getDataSize(quadrants[quadIdx].axes.x) * quadrants[quadIdx].entityWidth;
        quadrants[quadIdx].offsetY = (quadrantHeight - drawingHeight) / 2;
        quadrants[quadIdx].offsetX = (quadrantWidth  - drawingWidth)  / 2;
    }
}

// ==================================== PRIVATE FUNCTIONS  END  =============================================

// ==================================== PICKING RELATED CODE START ==========================================

function customMouseDown() {
    this.mouseDown = true;                                           // `this` is the canvas
}

function customMouseUp() {
    this.mouseDown = false;
}

/**
 * Implements picking and redraws the scene
 */
function customMouseMove(e) {
    if (!this.mouseDown)
        return;
    var selectedQuad = quadrants[Math.floor(e.offsetX / quadrantWidth)];
    // check if it's inside the quadrant but outside the drawing
    if (e.offsetY < selectedQuad.offsetY || e.offsetY >= quadrantHeight - selectedQuad.offsetY ||
        e.offsetX < quadrantWidth * selectedQuad.index + selectedQuad.offsetX ||
        e.offsetX >= quadrantWidth * (selectedQuad.index + 1) - selectedQuad.offsetX)
        return;
    var selectedEntityOnX = Math.floor((e.offsetX % quadrantWidth) / selectedQuad.entityWidth);
    var selectedEntityOnY = Math.floor((e.offsetY - selectedQuad.offsetY) / selectedQuad.entityHeight);

    selectedEntity[selectedQuad.axes.x] = selectedEntityOnX;
    selectedEntity[selectedQuad.axes.y] = selectedEntityOnY;
    drawScene()
}

// ==================================== PICKING RELATED CODE  END  ==========================================
