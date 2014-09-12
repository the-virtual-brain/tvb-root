// ==================================== INITIALIZATION CODE START ===========================================
var tsVol = {
    ctx: null,                  // The context for drawing on current canvas.
    currentQuadrant: 0,         // The quadrant we're in.
    quadrants: [],              // The quadrants array.
    minimumValue: null,         // Minimum value of the dataset.
    maximumValue: null,         // Maximum value of the dataset.
    voxelSize: null,
    volumeOrigin: null,         // TolumeOrigin is not used for now. if needed, use it in _setQuadrant
    selectedEntity: [0, 0, 0],  // The selected voxel; [i, j, k].
    selectedEntityValue: 0,     // he value of voxel[i,j,k]
    entitySize: [0, 0, 0],      // The size of each plane
    quadrantHeight: null,       // The height of the three small left quadrants
    quadrantWidth: null,        // The width of the three small left quadrants
    focusQuadrantHeight: null,  // The height of the focus quadrant
    focusQuadrantWidth: null,   // The width of the focus quadrant
    legendHeight: 0,            // The height of the legend quadrant
    legendWidth: 0,             // The width of the legend quadrant
    legendPadding:80*2,         // Horizontal padding for the TSV viewr legend
    selectedQuad: 0,            // The quadrant selected by the user every time
    highlightedQuad: {},        // The plane to be displayed on the focus quadrant
    timeLength: 0,              // Number of timepoints in the Volume.
    currentTimePoint: 0,
    playbackRate: 66,           // This is a not acurate lower limit for playback speed.
    playerIntervalID: null,     // ID from the player's setInterval().
    streamToBufferID: null,     // ID from the buffering system's setInterval().
    bufferSize: 1,              // How many time points do we load each time?
    bufferL2Size: 1,            // How many sets of buffers can we keep at the same time?
    lookAhead: 100,             // How many sets of buffers should be loaded ahead of us each time?
    data: {},                   // The actual data to be drawn to canvas.
    sliceArray: [],             // A helper variable to draw the data on the canvas.
    bufferL2: {},               // Contains all data loaded and preloaded, limited by memory.
    bufferL3: {},               // Contains all data from loaded views, limited by memory.
    urlMainData: "",            // Used to contain the python URL for each time point.
    urlVolumeData: "",          // Used to store the call for get_volume_view server function.
    dataSize: "",               // Used first to contain the file ID and then it's dimension.
    requestQueue: [],           // Used to avoid requesting a time point set while we are waiting for it.
    parserBlob: null,           // Used to store the JSON Parser Blob for web-workers.
    slidersClicked: false,      // Used to handle the status of the sliders.
    batchID: 0,                 // Used to ignore useless incoming ajax responses.
    urlTimeSeriesData: "",      // Contains the address to query the time series of a specific voxel.
    samplePeriod: 0,            // Meta data. The sampling period of the time series
    samplePeriodUnit: ""        // Meta data. The time unit of the sample period
};

var Quadrant = function (params){                // this keeps all necessary data for drawing
    this.index = params.index || 0;                // in a quadrant
    this.axes = params.axes || {x: 0, y: 1};       // axes represented in current quad; i=0, j=1, k=2
    this.entityWidth = params.entityWidth || 0;    // width and height of one voxel in current quad
    this.entityHeight = params.entityHeight || 0;
    this.offsetX = params.offsetX || 0;            // the offset of the drawing relative to the quad
    this.offsetY = params.offsetY || 0;
};

var SLIDERS = ["X", "Y", "Z"];
var SLIDERIDS = ["sliderForAxisX", "sliderForAxisY", "sliderForAxisZ"];

/**
 * Make all the necessary initialisations and draws the default view, with the center voxel selected
 * @param dataUrls          Urls containing data slices from server
 * @param minValue          The minimum value for all the slices
 * @param maxValue          The maximum value for all the slices
 * @param samplePeriod      Number representing how frequent the signal was being measured
 * @param samplePeriodUnit  Unit-Measure for the sampling period (e.g. ms, s)
 * @param volumeShape       Json with the shape of the full TS
 * @param volOrigin         The origin of the rendering; irrelevant in 2D, for now
 * @param sizeOfVoxel       How the voxel is sized on each axis; [xScale, yScale, zScale]
 */
function TSV_initVisualizer(dataUrls, minValue, maxValue, samplePeriod, samplePeriodUnit,
                            volumeShape, volOrigin, sizeOfVoxel) {

    var canvas = document.getElementById("canvasVolumes");
    if (!canvas.getContext){
        displayMessage('You need a browser with canvas capabilities, to see this demo fully!', "errorMessage");
        return;
    }

    /**
    * This will be our JSON parser web-worker blob,
    * Using a webworker is a bit slower than parsing the jsons with 
    * classical methods but it will prevent the main thread to be blocked 
    * while waiting for the parsing, granting a smooth visualization.
    * We use this technique also to avoid writing a separate file
    * for each worker.
    */
    tsVol.parserBlob = inlineWebWorkerWrapper(
            function(){
                self.addEventListener( 'message', function (e){
                            // Parse JSON, send it to main thread, close the worker
                            self.postMessage(JSON.parse(e.data));
                            self.close();
                }, false );
            }
        );

    tsVol.volumeOrigin = $.parseJSON(volOrigin)[0];
    tsVol.voxelSize    = $.parseJSON(sizeOfVoxel);

    canvas.height = $(canvas).parent().height();
    canvas.width  = $(canvas).parent().width();

    var tmp = canvas.height / 3;
    tsVol.quadrantHeight = tmp;       // quadrants are squares
    tsVol.quadrantWidth = tmp;
    tsVol.focusQuadrantWidth = canvas.width - tsVol.quadrantWidth;
    tsVol.legendHeight = canvas.height / 13;
    tsVol.legendWidth = tsVol.focusQuadrantWidth - tsVol.legendPadding;
    tsVol.focusQuadrantHeight = canvas.height - tsVol.legendHeight;

    tsVol.ctx = canvas.getContext("2d");
    // TODO maybe in the future we will find a solution to make image bigger before saving
    canvas.drawForImageExport = function() {};

    dataUrls = $.parseJSON(dataUrls);
    tsVol.urlMainData = dataUrls[0];
    tsVol.urlVolumeData = dataUrls[1];
    tsVol.urlTimeSeriesData = dataUrls[2];

    tsVol.dataSize = $.parseJSON(volumeShape);
    tsVol.samplePeriod = samplePeriod;
    tsVol.samplePeriodUnit = samplePeriodUnit;

    tsVol.minimumValue = minValue;
    tsVol.maximumValue = maxValue;
    tsVol.timeLength = tsVol.dataSize[0];           //Number of time points;

    _setupQuadrants();

    // set the center entity as the selected one
    tsVol.selectedEntity[0] = Math.floor(tsVol.dataSize[1] / 2); 
    tsVol.selectedEntity[1] = Math.floor(tsVol.dataSize[2] / 2);
    tsVol.selectedEntity[2] = Math.floor(tsVol.dataSize[3] / 2);

    // get entities number of voxels
    tsVol.entitySize[0] = tsVol.dataSize[1];
    tsVol.entitySize[1] = tsVol.dataSize[2];
    tsVol.entitySize[2] = tsVol.dataSize[3];
    // get entities number of time points
    tsVol.entitySize[3] = tsVol.timeLength;

    _setupBuffersSize();

    ColSch_initColorSchemeParams(minValue, maxValue, colorRedraw);
    tsVol.currentTimePoint = 0;
    // Update the data shared with the SVG Time Series Fragment
    updateTSFragment();

    // Start querying the server for volumetric data
    startBuffering();
    window.setInterval(freeBuffer, tsVol.playbackRate*10);

    // Start the SVG Time Series Fragment and draw it.
    TSF_initVisualizer(tsVol.urlTimeSeriesData);
    drawGraphs();
}
// ==================================== INITIALIZATION CODE END =============================================

// ==================================== DRAWING FUNCTIONS START =============================================

/**
 * Draws the current view depending on the selected entity
 * @param tIndex The time point we want to draw
 */
function drawSceneFunctional(tIndex){
    if(tsVol.playerIntervalID){
        drawSceneFunctionalFromView(tIndex)
    }
    else{
        drawSceneFunctionalFromCube(tIndex)
    }
    drawLegend();
    drawLabels();
}
/**
 * Update function necessary for the color picking function
 */
function colorRedraw(){
    drawSceneFunctional(tsVol.currentTimePoint);
}

/**
 * Draws the current scene from the whole loaded cube data
 * @param tIndex The time point we want to draw
 */
function drawSceneFunctionalFromCube(tIndex){
    var i, j, k, ii, jj, kk;

    // if we pass no tIndex the function will play
    // from the tsVol.currentTimePoint incrementing it by 1 or going back to 0.
    if(tIndex == null){
        tIndex = tsVol.currentTimePoint;
        tsVol.currentTimePoint++;
        tsVol.currentTimePoint = tsVol.currentTimePoint % tsVol.timeLength;
    }
    updateTSFragment();
    tsVol.data = getSliceAtTime(tIndex);
    _setCtxOnQuadrant(0);
    tsVol.ctx.fillStyle = ColSch_getGradientColorString(tsVol.minimumValue, tsVol.minimumValue, tsVol.maximumValue);
    tsVol.ctx.fillRect(0, 0, tsVol.ctx.canvas.width, tsVol.ctx.canvas.height);

    for (j = 0; j < tsVol.dataSize[2]; ++j)
        for (i = 0; i < tsVol.dataSize[1]; ++i)
            drawVoxel(i, j, tsVol.data[i][j][tsVol.selectedEntity[2]]);
    drawMargin();

    _setCtxOnQuadrant(1);
    for (k = 0; k < tsVol.dataSize[3]; ++k)
        for (jj = 0; jj < tsVol.dataSize[2]; ++jj)
            drawVoxel(k, jj, tsVol.data[tsVol.selectedEntity[0]][jj][k]);
    drawMargin();

    _setCtxOnQuadrant(2);
    for (kk = 0; kk < tsVol.dataSize[3]; ++kk)
        for (ii = 0; ii < tsVol.dataSize[1]; ++ii)
            drawVoxel(kk, ii, tsVol.data[ii][tsVol.selectedEntity[1]][kk]);
    drawMargin();
    drawFocusQuadrantFromCube(tIndex);
    drawNavigator();
    updateMoviePlayerSlider();
    setSelectedEntityValue(tsVol.data);
}

/**
 * Draws the selectedQuadrant on Focus Quadrant from the whole cube data
 * @param tIndex The time point we want to draw
 */
function drawFocusQuadrantFromCube(tIndex){
    _setCtxOnQuadrant(3);
    if(tsVol.highlightedQuad.index == 0){
        for (var j = 0; j < tsVol.dataSize[2]; ++j)
            for (var i = 0; i < tsVol.dataSize[1]; ++i)
                drawVoxel(i, j, tsVol.data[i][j][tsVol.selectedEntity[2]]);
    }
    else if(tsVol.highlightedQuad.index == 1){
        for (var k = 0; k < tsVol.dataSize[3]; ++k)
            for (var jj = 0; jj < tsVol.dataSize[2]; ++jj)
                drawVoxel(k, jj, tsVol.data[tsVol.selectedEntity[0]][jj][k]);
    }
    else if(tsVol.highlightedQuad.index == 2){
        for (var kk = 0; kk < tsVol.dataSize[3]; ++kk)
            for (var ii = 0; ii < tsVol.dataSize[1]; ++ii)
                drawVoxel(kk, ii, tsVol.data[ii][tsVol.selectedEntity[1]][kk]);
    }
    drawMargin();
}

/**
 * Draws the current scene only from the three visible planes data.
 * @param tIndex The time point we want to draw
 */
function drawSceneFunctionalFromView(tIndex){
    var i, j, k, ii, jj, kk;

    // if we pass no tIndex the function will play
    // from the tsVol.currentTimePoint incrementing it by 1 or going back to 0.
    if(tIndex == null){
        tIndex = tsVol.currentTimePoint;
        tsVol.currentTimePoint++;
        tsVol.currentTimePoint = tsVol.currentTimePoint % tsVol.timeLength;
    }
    updateTSFragment();
    // An array containing the view for each plane.
    tsVol.sliceArray = getViewAtTime(tIndex);

    _setCtxOnQuadrant(0);
    tsVol.ctx.fillStyle = ColSch_getGradientColorString(tsVol.minimumValue, tsVol.minimumValue, tsVol.maximumValue);
    tsVol.ctx.fillRect(0, 0, tsVol.ctx.canvas.width, tsVol.ctx.canvas.height);

    for (j = 0; j < tsVol.dataSize[2]; ++j)
        for (i = 0; i < tsVol.dataSize[1]; ++i)
            drawVoxel(i, j, tsVol.sliceArray[0][i][j])
    drawMargin();

    _setCtxOnQuadrant(1);
    for (k = 0; k < tsVol.dataSize[3]; ++k)
        for (jj = 0; jj < tsVol.dataSize[2]; ++jj)
            drawVoxel(k, jj, tsVol.sliceArray[1][jj][k])
    drawMargin();

    _setCtxOnQuadrant(2);
    for (kk = 0; kk < tsVol.dataSize[3]; ++kk)
        for (ii = 0; ii < tsVol.dataSize[1]; ++ii)
            drawVoxel(kk, ii, tsVol.sliceArray[2][ii][kk])

    drawMargin();
    drawFocusQuadrantFromView(tIndex);
    drawNavigator();
    updateMoviePlayerSlider();
    setSelectedEntityValue(tsVol.sliceArray);
}

/**
 * Draws the selectedQuadrant on Focus Quadrant from the xyz planes data.
 * @param tIndex The time point we want to draw
 */
function drawFocusQuadrantFromView(tIndex){
    _setCtxOnQuadrant(3);
    if(tsVol.highlightedQuad.index == 0){
        for (var j = 0; j < tsVol.dataSize[2]; ++j)
            for (var i = 0; i < tsVol.dataSize[1]; ++i)
                drawVoxel(i, j, tsVol.sliceArray[0][i][j]);
    }
    else if(tsVol.highlightedQuad.index == 1){
        for (var k = 0; k < tsVol.dataSize[3]; ++k)
            for (var jj = 0; jj < tsVol.dataSize[2]; ++jj)
                drawVoxel(k, jj, tsVol.sliceArray[1][jj][k]);
    }
    else if(tsVol.highlightedQuad.index == 2){
        for (var kk = 0; kk < tsVol.dataSize[3]; ++kk)
            for (var ii = 0; ii < tsVol.dataSize[1]; ++ii)
                drawVoxel(kk, ii, tsVol.sliceArray[2][ii][kk]);
    }
    drawMargin();
}

/**
 * Draws the voxel set at (line, col) in the current quadrant, and colors it
 * according to its value.
 * This function know nothing about the time point.
 * @param line THe vertical line of the grid we wish to draw on
 * @param col The horizontal line of the grid we wish to draw on
 * @param value The value of the voxel that will be converted into color
 */
function drawVoxel(line, col, value){
    tsVol.ctx.fillStyle = ColSch_getGradientColorString(value, tsVol.minimumValue, tsVol.maximumValue);
    // col increases horizontally and line vertically, so col represents the X drawing axis, and line the Y
	tsVol.ctx.fillRect(col * tsVol.currentQuadrant.entityWidth, line * tsVol.currentQuadrant.entityHeight,
	                   tsVol.currentQuadrant.entityWidth + 1, tsVol.currentQuadrant.entityHeight + 1);
}

/**
 * Draws the cross-hair on each quadrant, on the <code>tsVol.selectedEntity</code>
 */
function drawNavigator(){
    // Preview quadrans navigators
    tsVol.ctx.save();
    tsVol.ctx.beginPath();

    for (var quadIdx = 0; quadIdx < 3; ++quadIdx){
        _setCtxOnQuadrant(quadIdx);
        var x = tsVol.selectedEntity[tsVol.currentQuadrant.axes.x] * tsVol.currentQuadrant.entityWidth + tsVol.currentQuadrant.entityWidth / 2;
        var y = tsVol.selectedEntity[tsVol.currentQuadrant.axes.y] * tsVol.currentQuadrant.entityHeight + tsVol.currentQuadrant.entityHeight / 2;
        drawCrossHair(x, y);
    }
    tsVol.ctx.strokeStyle = "red";
    tsVol.ctx.lineWidth = 3;
    tsVol.ctx.stroke();
    tsVol.ctx.restore();

    // Focus quadrant Navigator
    tsVol.ctx.save();
    tsVol.ctx.beginPath();

    _setCtxOnQuadrant(3);
    var xx = tsVol.selectedEntity[tsVol.currentQuadrant.axes.x] * tsVol.currentQuadrant.entityWidth + tsVol.currentQuadrant.entityWidth / 2;
    var yy = tsVol.selectedEntity[tsVol.currentQuadrant.axes.y] * tsVol.currentQuadrant.entityHeight + tsVol.currentQuadrant.entityHeight / 2;
    drawFocusCrossHair(xx, yy);

    tsVol.ctx.strokeStyle = "blue";
    tsVol.ctx.lineWidth = 3;
    tsVol.ctx.stroke();
    tsVol.ctx.restore();
}

/**
 * Draws a 20px X 20px cross hair on the <code>tsVol.currentQuadrant</code>, at the specified x and y
 */
function drawCrossHair(x, y){
    tsVol.ctx.moveTo(Math.max(x - 20, 0), y);                              // the horizontal line
    tsVol.ctx.lineTo(Math.min(x + 20, tsVol.quadrantWidth), y);
    tsVol.ctx.moveTo(x, Math.max(y - 20, 0));                              // the vertical line
    tsVol.ctx.lineTo(x, Math.min(y + 20, tsVol.quadrantHeight));
}

/**
 * Draws a cross hair on the bigger focus quadrant, at the specified x and y
 */
function drawFocusCrossHair(x, y){
    tsVol.ctx.moveTo(Math.max(x - 20, 0), y);                              // the horizontal line
    tsVol.ctx.lineTo(Math.min(x + 20, tsVol.focusQuadrantWidth), y);
    tsVol.ctx.moveTo(x, Math.max(y - 20, 0));                              // the vertical line
    tsVol.ctx.lineTo(x, Math.min(y + 20, tsVol.focusQuadrantHeight));
}

/**
 * Draws a canvas legend at the bottom of the volume visualizer:
 * <ul>
 * <li>Written min, mean and max values for the data, in scientific notation.
 * <li>Displays the color range for the data.
 * <li>Displays a white bar in to show the currently selected entity value.
 * </ul>
 */
function drawLegend(){
    var tmp = tsVol.legendPadding / 2;
    // set the context on the legend quadrant
    tsVol.ctx.setTransform(1, 0, 0, 1, 0, 0);  // reset the transformation
    tsVol.ctx.translate( tsVol.quadrantWidth + tmp, tsVol.focusQuadrantHeight);
    // set the font properties
    tsVol.ctx.font = '12px Helvetica';
    tsVol.ctx.textAlign = 'center';
    tsVol.ctx.textBaseline = 'middle';
    tsVol.ctx.fillStyle = 'white';
    // write min, mean, max values on the canvas
    tsVol.ctx.fillText(tsVol.minimumValue.toExponential(2), 0, tsVol.legendHeight - 10);
    tsVol.ctx.fillText("|", 1, tsVol.legendHeight/2);
    var midValue = (tsVol.maximumValue - tsVol.minimumValue)/2;
    tsVol.ctx.fillText(midValue.toExponential(2), tsVol.legendWidth/2, tsVol.legendHeight - 10);
    tsVol.ctx.fillText("|", tsVol.legendWidth/2, tsVol.legendHeight/2);
    tsVol.ctx.fillText(tsVol.maximumValue.toExponential(2), tsVol.legendWidth, tsVol.legendHeight - 10);
    tsVol.ctx.fillText("|", tsVol.legendWidth-1, tsVol.legendHeight/2);

    // Draw a color bar from min to max value based on the selected color coding
    for(var i = 0; i< tsVol.legendWidth; i++){
        var val = tsVol.minimumValue + ((i/tsVol.legendWidth)*(tsVol.maximumValue-tsVol.minimumValue));
        tsVol.ctx.fillStyle = ColSch_getGradientColorString(val, tsVol.minimumValue, tsVol.maximumValue);
        tsVol.ctx.fillRect(i, 1, 1.5, tsVol.legendHeight/2);
    }

    // Draw the selected entity value marker on the color bar
    tsVol.ctx.fillStyle = "white";
    tmp = tsVol.selectedEntityValue;
    tmp = tmp / (tsVol.maximumValue - tsVol.minimumValue);
    tmp = (tmp * tsVol.legendWidth);
    tsVol.ctx.fillRect(tmp, 1, 3, tsVol.legendHeight/2);
}

/**
 * Add spatial labels to the navigation quadrants
 */
function drawLabels(){
    tsVol.ctx.font = '15px Helvetica';
    tsVol.ctx.textBaseline = 'middle';
    _setCtxOnQuadrant(0);
    tsVol.ctx.fillText("Axial", tsVol.quadrantWidth/5, tsVol.quadrantHeight - 13);
    _setCtxOnQuadrant(1);
    tsVol.ctx.fillText("Sagittal", tsVol.quadrantWidth/5, tsVol.quadrantHeight - 13);
    _setCtxOnQuadrant(2);
    tsVol.ctx.fillText("Coronal", tsVol.quadrantWidth/5, tsVol.quadrantHeight - 13);
}

/**
 * Draws a 5px rectangle around the <code>tsVol.currentQuadrant</code>
 */
function drawMargin(){
    var marginWidth, marginHeight;
    if(tsVol.currentQuadrant.index == 3){
        marginWidth = tsVol.focusQuadrantWidth;
        marginHeight = tsVol.focusQuadrantHeight;
    }
    else{
        marginWidth = tsVol.quadrantWidth;
        marginHeight = tsVol.quadrantHeight;
    }
    tsVol.ctx.beginPath();
    tsVol.ctx.rect(2, 0, marginWidth - 3, marginHeight - 2);
    tsVol.ctx.lineWidth = 2;
    if(tsVol.currentQuadrant.index == tsVol.selectedQuad.index && tsVol.currentQuadrant.index != 3){
        tsVol.ctx.strokeStyle = 'white';
        tsVol.highlightedQuad = tsVol.currentQuadrant;
    }
    else if(tsVol.currentQuadrant.index == tsVol.highlightedQuad.index && tsVol.selectedQuad.index == 3){
        tsVol.ctx.strokeStyle = 'white';
    }
    else{
        tsVol.ctx.strokeStyle = 'gray';
    }
    tsVol.ctx.stroke();
}

// ==================================== DRAWING FUNCTIONS  END  =============================================

// ==================================== PRIVATE FUNCTIONS START =============================================

/**
 * Sets the <code>tsVol.currentQuadrant</code> and applies transformations on context depending on that
 *
 * @param quadIdx Specifies which of <code>quadrants</code> is selected
 * @private
 */
/* TODO: make it use volumeOrigin; could be like this:
 * <code>tsVol.ctx.setTransform(1, 0, 0, 1, volumeOrigin[tsVol.currentQuadrant.axes.x], volumeOrigin[tsVol.currentQuadrant.axes.y])</code>
 *       if implemented, also change the picking to take it into account
 */
function _setCtxOnQuadrant(quadIdx){
    tsVol.currentQuadrant = tsVol.quadrants[quadIdx];
    tsVol.ctx.setTransform(1, 0, 0, 1, 0, 0);                              // reset the transformation
    // Horizontal Mode
    //tsVol.ctx.translate(quadIdx * tsVol.quadrantWidth + tsVol.currentQuadrant.offsetX, tsVol.currentQuadrant.offsetY);
    // Vertical Mode
    if(quadIdx == 3){
       tsVol.ctx.translate(tsVol.quadrantWidth + tsVol.currentQuadrant.offsetX, tsVol.currentQuadrant.offsetY);
    }
    else{
        tsVol.ctx.translate(tsVol.currentQuadrant.offsetX, quadIdx * tsVol.quadrantHeight +  tsVol.currentQuadrant.offsetY);
    }
}

/**
 * Returns the number of elements on the given axis
 * @param axis The axis whose length is returned; i=0, j=1, k=2
 * @returns {*}
 * @private
 */
function _getDataSize(axis){
    switch (axis){
        case 0:     return tsVol.dataSize[1];
        case 1:     return tsVol.dataSize[2];
        case 2:     return tsVol.dataSize[3];
    }
}

/**
 * Computes the actual dimension of one entity from the specified axes
 * @param xAxis The axis to be represented on X (i=0, j=1, k=2)
 * @param yAxis The axis to be represented on Y (i=0, j=1, k=2)
 * @returns {{width: number, height: number}} Entity width and height
 */
function _getEntityDimensions(xAxis, yAxis){
    var scaleOnWidth  = tsVol.quadrantWidth  / (_getDataSize(xAxis) * tsVol.voxelSize[xAxis]);
    var scaleOnHeight = tsVol.quadrantHeight / (_getDataSize(yAxis) * tsVol.voxelSize[yAxis]);
    var scale = Math.min(scaleOnHeight, scaleOnWidth);
    return {width: tsVol.voxelSize[xAxis] * scale, height: tsVol.voxelSize[yAxis] * scale}
}

/**
 * Computes the actual dimension of one entity from the specified axes
 * To be used to set the dimensions of data on the "focus" Quadrant
 * @param xAxis The axis to be represented on X (i=0, j=1, k=2)
 * @param yAxis The axis to be represented on Y (i=0, j=1, k=2)
 * @returns {{width: number, height: number}} Entity width and height
 */
function _getFocusEntityDimensions(xAxis, yAxis){
    var scaleOnWidth  = tsVol.focusQuadrantWidth  / (_getDataSize(xAxis) * tsVol.voxelSize[xAxis]);
    var scaleOnHeight = tsVol.focusQuadrantHeight / (_getDataSize(yAxis) * tsVol.voxelSize[yAxis]);
    var scale = Math.min(scaleOnHeight, scaleOnWidth);
    return {width: tsVol.voxelSize[xAxis] * scale, height: tsVol.voxelSize[yAxis] * scale}
}

/**
 * Initializes the <code>tsVol.quadrants</code> with some default axes and sets their properties
 */
function _setupQuadrants(){
    tsVol.quadrants.push(new Quadrant({ index: 0, axes: {x: 1, y: 0} }));
    tsVol.quadrants.push(new Quadrant({ index: 1, axes: {x: 1, y: 2} }));
    tsVol.quadrants.push(new Quadrant({ index: 2, axes: {x: 0, y: 2} }));

    for (var quadIdx = 0; quadIdx < tsVol.quadrants.length; quadIdx++){
        var entityDimensions = _getEntityDimensions(tsVol.quadrants[quadIdx].axes.x, tsVol.quadrants[quadIdx].axes.y);
        tsVol.quadrants[quadIdx].entityHeight = entityDimensions.height;
        tsVol.quadrants[quadIdx].entityWidth  = entityDimensions.width;
        tsVol.quadrants[quadIdx].offsetY = 0;
        tsVol.quadrants[quadIdx].offsetX = 0;
    }
    tsVol.selectedQuad = tsVol.quadrants[0];
    tsVol.highlightedQuad = tsVol.selectedQuad;
    _setupFocusQuadrant();
}

/**
 * Helper function to setup and add the Focus Quadrant to <code>tsVol.quadrants</code>.
 */
function _setupFocusQuadrant(){
    if(tsVol.quadrants.length == 4){
        tsVol.quadrants.pop();
    }
    var axe = 0;
    if(tsVol.selectedQuad.index == 0){
        axe = {x: 1, y: 0};
    }
    else if(tsVol.selectedQuad.index == 1){
        axe = {x: 1, y: 2};
    }
    else{
        axe = {x: 0, y: 2};
    }
    tsVol.quadrants.push(new Quadrant({ index: 3, axes: axe }));
    var entityDimensions = _getFocusEntityDimensions(tsVol.quadrants[3].axes.x, tsVol.quadrants[3].axes.y);
    tsVol.quadrants[3].entityHeight = entityDimensions.height;
    tsVol.quadrants[3].entityWidth  = entityDimensions.width;
    tsVol.quadrants[3].offsetY = 0;
    tsVol.quadrants[3].offsetX = 0;
}

/**
 * Automatically determine optimal bufferSizer, depending on data dimensions.
 */
function _setupBuffersSize() {
    var tpSize = Math.max(tsVol.entitySize[0], tsVol.entitySize[1], tsVol.entitySize[2]);
    tpSize = tpSize * tpSize;
    //enough to avoid waisting bandwidth and to parse the json smoothly
    while(tsVol.bufferSize * tpSize <= 50000){
        tsVol.bufferSize++;
    }
    //Very safe measure to avoid crashes. Tested on Chrome.
    while(tsVol.bufferSize * tpSize * tsVol.bufferL2Size <= 157286400){
        tsVol.bufferL2Size *= 2;
    }
    tsVol.bufferL2Size /= 2;
}

// ==================================== PRIVATE FUNCTIONS  END  =============================================

// ==================================== HELPER FUNCTIONS START ==============================================

/**
 * Requests file data without blocking the main thread if possible.
 * @param fileName The target URL or our request
 * @param sect The section index of the wanted data in our buffering system.
 */
function asyncRequest(fileName, sect){
    var index = tsVol.requestQueue.indexOf(sect);
    var privateID = tsVol.batchID;

    if (index < 0){
        tsVol.requestQueue.push(sect);
        doAjaxCall({
            async:true,
            url:fileName,
            method:"POST",
            mimeType:"text/plain",
            success:function(response){
                if(privateID == tsVol.batchID){
                    parseAsync(response, function(json){
                        // save the parsed JSON
                        tsVol.bufferL3[sect] = json;
                        var idx = tsVol.requestQueue.indexOf(sect);
                        if (idx > -1){
                            tsVol.requestQueue.splice(idx, 1);
                        }
                    });
                }
            },
            error: function(){
                displayMessage("Could not retrieve data from the server!", "warningMessage");
            }
        });
    }
}

/**
 * Build a worker from an anonymous function body. Returns and URL Blob
 * @param workerBody The anonymous function to convert into URL BLOB
 * @returns URL Blob that can be used to invoke a web worker
 */
function inlineWebWorkerWrapper(workerBody){
    return URL.createObjectURL(
        new Blob(['(', workerBody.toString(), ')()' ], { type: 'application/javascript' }
        )
    );
}

/**
 * Parses JSON data in a web-worker. Has a fall back to traditional parsing.
 * @param data The json data to be parsed
 * @param callback Function to be called after the parsing
 */
function parseAsync(data, callback){
    var worker;
    var json;
    if( window.Worker ){
        worker = new Worker( tsVol.parserBlob );
        worker.addEventListener( 'message', function (e){
            json = e.data;
            callback( json );
        }, false);
        worker.postMessage( data );
    }
    else{
        json = JSON.parse( data );
        callback( json );
    }
}

/**
 *  This function is called whenever we can, to load some data ahead of
 *  were we're looking.
 */
function streamToBuffer(){
    // we avoid having too many requests at the same time
    if(tsVol.requestQueue.length < 2){
        var currentSection = Math.floor(tsVol.currentTimePoint/tsVol.bufferSize);
        var maxSections = Math.floor(tsVol.timeLength/tsVol.bufferSize);
        var xPlane = ";x_plane=" + (tsVol.selectedEntity[0]);
        var yPlane = ";y_plane=" + (tsVol.selectedEntity[1]);
        var zPlane = ";z_plane=" + (tsVol.selectedEntity[2]);

        for( var i = 0; i <= tsVol.lookAhead; i++ ){
            var toBufferSection = Math.min( currentSection + i, maxSections );
            if(!tsVol.bufferL3[toBufferSection] && tsVol.requestQueue.indexOf(toBufferSection) < 0){
                var from = toBufferSection*tsVol.bufferSize;
                var to = Math.min(from+tsVol.bufferSize, tsVol.timeLength);
                from = "from_idx=" + from;
                to = ";to_idx=" + to;
                var query = tsVol.urlVolumeData + from + to + xPlane + yPlane + zPlane;
                asyncRequest(query, toBufferSection);
                return; // break out of the loop
            }
        }
    }
}

/**
 *  This function is called to erase some elements from bufferL3 array and avoid
 *  consuming too much memory.
 */
function freeBuffer(){
    var section = Math.floor(tsVol.currentTimePoint/tsVol.bufferSize);
    var bufferedElements = Object.keys(tsVol.bufferL3).length;
    if(bufferedElements > tsVol.bufferL2Size){
        tsVol.bufferL2 = {};
        for(var idx in tsVol.bufferL3){
            if (idx < (section - tsVol.bufferL2Size/2) % tsVol.timeLength || idx > (section + tsVol.bufferL2Size/2) % tsVol.timeLength) {
                delete tsVol.bufferL3[idx];
            }
        }
    }
}

/**
 * A helper function similar to python's range().
 * @param len Integer
 * @returns An array with all integers from 0 to len-1
 */
function range(len){
    return Array.apply(null, new Array(len)).map(function (_, i){return i;});
}

/**
 * This functions returns the whole x,y,z cube data at time-point t.
 * @param t The time piont we want to get
 * @returns The whole x,y,z array data at time-point t.
 */
function getSliceAtTime(t){
    var buffer;
    var from = "from_idx=" + t;
    var to = ";to_idx=" + (t +1);
    var query = tsVol.urlMainData + from + to;

    if(tsVol.bufferL2[t]){
        buffer = tsVol.bufferL2[t];
    }else{
        tsVol.bufferL2[t] = HLPR_readJSONfromFile(query);
        buffer = tsVol.bufferL2[t];
    }
    return buffer[0];
}

/**
 *  This functions returns the X,Y,Z data from time-point t.
 * @param t The time piont we want to get
 * @returns Array whith only the data from the x,y,z plane at time-point t.
 */
function getViewAtTime(t){
    var buffer;
    var from;
    var to;
    var xPlane = ";x_plane=" + (tsVol.selectedEntity[0]);
    var yPlane = ";y_plane=" + (tsVol.selectedEntity[1]);
    var zPlane = ";z_plane=" + (tsVol.selectedEntity[2]);

    var query;

    var section = Math.floor(t/tsVol.bufferSize);

    if(tsVol.bufferL3[section]){ // We have that slice in memory
        buffer = tsVol.bufferL3[section];
    }else{ // We need to load that slice from the server
        from = "from_idx=" + t;
        to = ";to_idx=" + Math.min(1 + t, tsVol.timeLength);
        query = tsVol.urlVolumeData + from + to + xPlane + yPlane + zPlane;

        buffer = HLPR_readJSONfromFile(query);
        return [buffer[0][0],buffer[1][0],buffer[2][0]];
    }
    t = t%tsVol.bufferSize;
    return [buffer[0][t],buffer[1][t],buffer[2][t]];
}

/**
 * Sets tsVol.selectedEntityValue, wich represents the selected voxel value
 * @param data A multi dimensional array containing the relevant data
 */
function setSelectedEntityValue(data){
    // data comes from plane View
    if(data.length == 3){
        tsVol.selectedEntityValue = data[0][tsVol.selectedEntity[0]][tsVol.selectedEntity[1]];
    }
    // data comes from Cube
    else{
        tsVol.selectedEntityValue = data[tsVol.selectedEntity[0]][tsVol.selectedEntity[1]][tsVol.selectedEntity[2]];
    }
}

// ==================================== HELPER FUNCTIONS END ==============================================

// ==================================== PICKING RELATED CODE START ==========================================

function customMouseDown(e){
    e.preventDefault();
    this.mouseDown = true;            // `this` is the canvas
    TSV_pick(e)
}

function customMouseUp(e){
    e.preventDefault();
    this.mouseDown = false;
    startBuffering();
    if(tsVol.resumePlayer){
        window.setTimeout(playBack, tsVol.playbackRate*5);
        tsVol.resumePlayer = false;
    }
    if(tsVol.selectedQuad.index == 3){
        drawGraphs();
    }
}

function customMouseMove(e){
    if (!this.mouseDown){
        return;
    }
    e.preventDefault();
    TSV_pick(e);
}

/**
 * Implements picking and redraws the scene. Updates sliders too.
 * @param e The click event
 */
function TSV_pick(e){
    tsVol.bufferL3 = {};
    stopBuffering();
    if(tsVol.playerIntervalID){
        stopPlayback();
        tsVol.resumePlayer = true;
    }
    //fix for Firefox
    var offset = $('#canvasVolumes').offset();
    var xpos = e.pageX - offset.left;
    var ypos = e.pageY - offset.top;
    //var selectedQuad = tsVol.quadrants[Math.floor(xpos / tsVol.quadrantWidth)];
    if(Math.floor(xpos / tsVol.quadrantWidth) >= 1){
        tsVol.selectedQuad = tsVol.quadrants[3];
        // check if it's inside the focus quadrant but outside the drawing
        if (ypos < tsVol.selectedQuad.offsetY ){
            return;
        }
        else if(ypos >= tsVol.focusQuadrantHeight - tsVol.selectedQuad.offsetY){
            return;
        }
        else if(xpos < tsVol.offsetX){
            return;
        }
        else if(xpos >= tsVol.focusQuadrantWidth - tsVol.selectedQuad.offsetX + tsVol.quadrantWidth){
            return;
        }
    } else{
        tsVol.selectedQuad = tsVol.quadrants[Math.floor(ypos / tsVol.quadrantHeight)];
        _setupFocusQuadrant();
        // check if it's inside the quadrant but outside the drawing
        if (ypos < tsVol.selectedQuad.offsetY ){
            return;
        }
        else if(ypos >= tsVol.quadrantHeight * (tsVol.selectedQuad.index + 1) - tsVol.selectedQuad.offsetY){
            return;
        }
        else if(xpos < tsVol.offsetX){
            return;
        }
        else if(xpos >= tsVol.quadrantWidth - tsVol.selectedQuad.offsetX){
            return;
        }
    }

    var selectedEntityOnX = 0;
    var selectedEntityOnY = 0;

    if(tsVol.selectedQuad.index == 3){
        selectedEntityOnX = Math.floor(((xpos - tsVol.quadrantWidth) % tsVol.focusQuadrantWidth) / tsVol.selectedQuad.entityWidth);
        selectedEntityOnY = Math.floor(((ypos - tsVol.selectedQuad.offsetY) % tsVol.focusQuadrantHeight) / tsVol.selectedQuad.entityHeight);
    } else{
        selectedEntityOnX = Math.floor((xpos - tsVol.selectedQuad.offsetX) / tsVol.selectedQuad.entityWidth);
        selectedEntityOnY = Math.floor((ypos % tsVol.quadrantHeight) / tsVol.selectedQuad.entityHeight);
    }
    tsVol.selectedEntity[tsVol.selectedQuad.axes.x] = selectedEntityOnX;
    tsVol.selectedEntity[tsVol.selectedQuad.axes.y] = selectedEntityOnY;
    updateTSFragment();
    updateSliders();
    drawSceneFunctional(tsVol.currentTimePoint);
}

// ==================================== PICKING RELATED CODE  END  ==========================================

// ==================================== UI RELATED CODE START ===============================================
/**
 * Functions that calls all the setup functions for the main UI of the time series volume visualizer.
 */
function TSV_startUserInterface() {
    startPositionSliders();
    startMovieSlider();
}

/**
 * Code for the navigation slider. Creates the x,y,z sliders and adds labels
 */
function startPositionSliders() {

    // We loop trough every slider
    for(var i = 0; i < 3; i++) {
        $("#sliderForAxis" + SLIDERS[i]).each(function() {
            var value = tsVol.selectedEntity[i];
            var opts = {
                            value: value,
                            min: 0,
                            max: tsVol.entitySize[i] - 1, // yeah.. if we start from zero we need to subtract 1
                            animate: true,
                            orientation: "horizontal",
                            change: slideMoved, // call this function *after* the slide is moved OR the value changes
                            slide: slideMove  //  we use this to keep it smooth.
                        };
            $(this).slider(opts);
            $("#labelCurrentValueAxis" + SLIDERS[i]).empty().text("[" + value + "]");
            $("#labelMaxValueAxis" + SLIDERS[i]).empty().text(opts.max);
        });
    }
}

/**
 * Code for "movie player" slider. Creates the slider and adds labels
 */
function startMovieSlider(){
    $("#movieSlider").each(function() {
        var value = 0;
        var opts = {
                    value: value,
                    min: 0,
                    max: tsVol.timeLength - 1,
                    animate: true,
                    orientation: "horizontal",
                    range: "min",
                    stop: moviePlayerMoveEnd,
                    slide: moviePlayerMove
                    };
        $(this).slider(opts);

        var actualTime = value * tsVol.samplePeriod;
        var totalTime = (tsVol.timeLength - 1) * tsVol.samplePeriod;
        $("#labelCurrentTimeStep").empty().text("[" + actualTime.toFixed(2)+ "]");
        $("#labelMaxTimeStep").empty().text(totalTime.toFixed(2) + " ("+ tsVol.samplePeriodUnit + ")");
    });
}

// ==================================== CALLBACK FUNCTIONS START ===============================================

function playBack(){
    if(!tsVol.playerIntervalID)
        tsVol.playerIntervalID = window.setInterval(drawSceneFunctional, tsVol.playbackRate);
        $("#btnPlay")[0].setAttribute("class", "action action-pause");
}

function stopPlayback(){
    window.clearInterval(tsVol.playerIntervalID);
    tsVol.playerIntervalID = null;
    $("#btnPlay")[0].setAttribute("class", "action action-run");
}

function togglePlayback() {
    if(!tsVol.playerIntervalID) {
        playBack();
    } else {
        stopPlayback();
    }
}

function startBuffering(){
    if(!tsVol.streamToBufferID){
        tsVol.batchID++;
        tsVol.requestQueue = [];
        tsVol.bufferL3 = {};
        tsVol.streamToBufferID = window.setInterval(streamToBuffer, 0);
    }
}

function stopBuffering(){
    window.clearInterval(tsVol.streamToBufferID);
    tsVol.streamToBufferID = null;
    tsVol.batchID++;
    tsVol.requestQueue = [];
    tsVol.bufferL3 = {};
}

function playNextTimePoint(){
    tsVol.currentTimePoint++;
    tsVol.currentTimePoint = tsVol.currentTimePoint%(tsVol.timeLength);
    drawSceneFunctionalFromView(tsVol.currentTimePoint);
    drawLegend();
    drawLabels();
}

function playPreviousTimePoint(){
    if(tsVol.currentTimePoint === 0)
        tsVol.currentTimePoint = tsVol.timeLength;
    drawSceneFunctionalFromView(--tsVol.currentTimePoint);
    drawLegend();
    drawLabels();
}

function seekFirst(){
    tsVol.currentTimePoint = 0;
    drawSceneFunctionalFromView(tsVol.currentTimePoint);
    drawLegend();
    drawLabels();
}

function seekEnd(){
    tsVol.currentTimePoint = tsVol.timeLength - 1;
    drawSceneFunctionalFromView(tsVol.currentTimePoint - 1);
    drawLegend();
    drawLabels();
}

/**
 * Updates the position and values of the x,y,z navigation sliders when we click the canvas.
 */
function updateSliders() {
    for(var i = 0; i < 3; i++) {
        $("#sliderForAxis" + SLIDERS[i]).each(function() {
            $(this).slider("option", "value", tsVol.selectedEntity[i]);                 //Update the slider value
        });
        $('#labelCurrentValueAxis' + SLIDERS[i]).empty().text('[' + tsVol.selectedEntity[i] + ']' ); //update label
    }
}

/**
 * While the navigation sliders are moved, this redraws the scene accordingly.
 */
function slideMove(event, ui){
    tsVol.bufferL3 = {};
    stopBuffering();
    if(tsVol.playerIntervalID){
        stopPlayback();
        tsVol.resumePlayer = true;
    }
    tsVol.slidersClicked = true;
    _coreMoveSliderAxis(event, ui);
    drawSceneFunctional(tsVol.currentTimePoint);
}

/**
 * After the navigation sliders are changed, this redraws the scene accordingly.
 */
function slideMoved(event, ui){
    if(tsVol.slidersClicked){
        startBuffering();
        tsVol.slidersClicked = false;

        if(tsVol.resumePlayer){
            window.setTimeout(playBack, tsVol.playbackRate * 5);
            tsVol.resumePlayer = false;
        }
    }
   _coreMoveSliderAxis(event, ui);
}

function _coreMoveSliderAxis(event, ui) {
    var quadID = SLIDERIDS.indexOf(event.target.id);
    var selectedQuad = tsVol.quadrants[quadID];

    //  Updates the label value on the slider.
    $("#labelCurrentValueAxis" + SLIDERS[quadID]).empty().text( '[' + ui.value + ']' );
    //  Setup things to draw the scene pointing to the right voxel and redraw it.
    if(quadID == 1) {
        tsVol.selectedEntity[selectedQuad.axes.x] = ui.value;
    } else {
        tsVol.selectedEntity[selectedQuad.axes.y] = ui.value;
    }
    updateTSFragment();
}

/**
 * Updated the player slider bar while playback is on.
 */
function updateMoviePlayerSlider() {
    _coreUpdateMovieSlider(tsVol.currentTimePoint, true);
}

/**
 * Updates the value at the end of the player bar when we move the handle.
 */
function moviePlayerMove(event, ui) {
    _coreUpdateMovieSlider(ui.value, false);
}

function _coreUpdateMovieSlider(timePoint, updateSlider) {
    $("#movieSlider").each(function() {
        if (updateSlider) {
            $(this).slider("option", "value", tsVol.currentTimePoint);
        }
        var actualTime = timePoint * tsVol.samplePeriod;
        $('#labelCurrentTimeStep').empty().text("[" + actualTime.toFixed(2) + "]");
    });
    d3.select(".timeVerticalLine").attr("transform", function(){
                var width = $(".graph-timeSeries-rect").attr("width");
                var pos = (timePoint * width) / (tsVol.timeLength);
                var bMin = Math.max(0, timePoint - 30);
                var bMax = Math.min(timePoint + 30, tsVol.timeLength);
                d3.select('.brush').transition()
                  .delay(0)
                  .call(tsFrag.brush.extent([bMin, bMax]))
                  .call(tsFrag.brush.event);
                return "translate(" + pos + ", 0)";
            });
}

/*
* Redraws the scene at the selected time-point at the end of a slide action.
* Calling this during the whole slide showed to be too expensive, so the
* new time-point is drawn only when the user releases the click from the handler
*/
function moviePlayerMoveEnd(event, ui){
    tsVol.currentTimePoint = ui.value;
    drawSceneFunctionalFromView(tsVol.currentTimePoint);
    drawLegend();
    drawLabels();
}

// ==================================== CALLBACK FUNCTIONS END ===============================================
// ==================================== UI RELATED CODE END ==================================================
