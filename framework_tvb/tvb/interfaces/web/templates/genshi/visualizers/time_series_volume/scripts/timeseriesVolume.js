
var tsVol = { 
    ctx: null,                  // the context for drawing on current canvas.
    currentQuadrant: 0,         // the quadrant we're in.
    quadrants: [],              // the quadrants array.
    minimumValue: null,         // minimum value of the dataset.
    maximumValue: null,         // maximum value of the dataset.
    voxelSize: null,
    volumeOrigin: null,         // volumeOrigin is not used for now. if needed, use it in _setQuadrant
    selectedEntity: [0, 0, 0],  // the selected voxel; [i, j, k].
    entitySize: [0, 0, 0],
    quadrantHeight: null,
    quadrantWidth: null,
    timeLength: 0,              // number of timepoints in the Volume.
    currentTimePoint: 0,
    playbackRate: 100,          // This is a not acurate lower limit for playback speed.
    playerIntervalID: null,     // ID from the player's setInterval().
    bufferSize: 1,              // How many time points do we load each time?
    bufferL2Size: 1,            // How many sets of buffers can we keep at the same time?
    lookAhead: 5,               // How many sets of buffers should be loaded ahead of us each time?
    data: {},                   // The actual data to be drawn to canvas.
    bufferL2: {},               // Cotains all data loaded and preloaded, limited by memory.
    dataAddress: "",            // Used to contain the python URL for each time point.
    dataView: "",               // Used to store the call for get_volume_view server function.
    dataSize: "",               // Used first to contain the file ID and then it's dimension.
    requestQueue: [],           // Used to avoid requesting a time point set while we are waiting for it.
    parserBlob: null            // Used to store the JSON Parser Blob for web-workers.
};

var Quadrant = function (params) {                // this keeps all necessary data for drawing
    this.index = params.index || 0;                // in a quadrant
    this.axes = params.axes || {x: 0, y: 1};       // axes represented in current quad; i=0, j=1, k=2
    this.entityWidth = params.entityWidth || 0;    // width and height of one voxel in current quad
    this.entityHeight = params.entityHeight || 0;
    this.offsetX = params.offsetX || 0;            // the offset of the drawing relative to the quad
    this.offsetY = params.offsetY || 0;
};

/**
 * Make all the necessary initialisations and draws the default view, with the center voxel selected
 * @param dataUrls  Urls containing data slices from server
 * @param minValue  The minimum value for all the slices
 * @param maxValue  The maximum value for all the slices
 * @param volOrigin The origin of the rendering; irrelevant in 2D, for now
 * @param sizeOfVoxel   How the voxel is sized on each axis; [xScale, yScale, zScale]
 */
function TSV_initVisualizer(dataUrls, minValue, maxValue, volOrigin, sizeOfVoxel) {
    var canvas = document.getElementById("volumetric-ts-canvas");
    if (!canvas.getContext) {
        displayMessage('You need a browser with canvas capabilities, to see this demo fully!', "errorMessage");
        return;
    }

    // This will be our JSON parser web-worker blob
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

    var canvasWidth = $(canvas).parent().width();
    canvas.width  = canvasWidth;                          // fill the screen on width
    canvas.height = canvasWidth / 3;                      // three quadrants + some space for labeling
    tsVol.quadrantHeight = canvasWidth / 3;               // quadrants are squares
    tsVol.quadrantWidth = canvasWidth/ 3;

    tsVol.ctx = canvas.getContext("2d");
    tsVol.ctx.beginPath();
    tsVol.ctx.rect(-2, -2, canvas.width, canvas.height);
    tsVol.ctx.lineWidth = 2;
    tsVol.ctx.strokeStyle = 'black';
    tsVol.ctx.stroke();

    dataUrls = $.parseJSON(dataUrls);
    tsVol.dataAddress = dataUrls[0];
    tsVol.dataView = dataUrls[2];
    tsVol.dataSize = HLPR_readJSONfromFile(dataUrls[1]);

    var query = tsVol.dataAddress+"from_idx="+(0)+";to_idx="+(1);
    tsVol.data = HLPR_readJSONfromFile(query);

    tsVol.minimumValue = minValue;
    tsVol.maximumValue = maxValue;
    tsVol.timeLength = tsVol.dataSize[0]; //Number of time points;

    _setupQuadrants();

    tsVol.selectedEntity[0] = Math.floor(tsVol.dataSize[1] / 2);       // set the center entity as the selected one
    tsVol.selectedEntity[1] = Math.floor(tsVol.dataSize[2] / 2);
    tsVol.selectedEntity[2] = Math.floor(tsVol.dataSize[3] / 2);

    tsVol.entitySize[0] = tsVol.dataSize[1];           // get entities number of voxels
    tsVol.entitySize[1] = tsVol.dataSize[2];
    tsVol.entitySize[2] = tsVol.dataSize[3];
    tsVol.entitySize[3] = tsVol.dataSize[0];           // get entities number of time points

    _setupBuffersSize();

    ColSch_initColorSchemeParams(minValue, maxValue, drawSceneFunctional);

    tsVol.data = getSliceAtTime(tsVol.currentTimePoint);
    drawSceneFunctional(tsVol.currentTimePoint);

    //window.setInterval(streamToBuffer, tsVol.playbackRate);
    //window.setInterval(freeBuffer, tsVol.playbackRate*10);
}

/**
 * Requests file data not blocking the main thread if possible.
 */
function asyncRequest(fileName, sect) {
    var start = new Date().getTime();
    tsVol.requestQueue.push(sect);
    console.log("tsVol.requestQueue push:", tsVol.requestQueue);
    doAjaxCall({
        async:true,
        url:fileName,
        methos:"GET",
        mimeType:"text/plain",
        success:function(r){
            parseAsync(r, function(json){
                tsVol.bufferL2[sect] = json;
                var index = tsVol.requestQueue.indexOf(sect);
                if (index > -1) {
                    tsVol.requestQueue.splice(index, 1);
                    console.log("tsVol.requestQueue pop", tsVol.requestQueue);
                    var end = new Date().getTime();
                    var time = end - start;
                    console.log("ASYNC took:", time);
                }   
            });
        }
    });
}

/**
 * Build a worker from an anonymous function body. Returns and URL Blob
 */
function inlineWebWorkerWrapper(workerBody){
    var retBlob = URL.createObjectURL(
        new Blob([
            '(',
                workerBody.toString(),
            ')()' ],
        { type: 'application/javascript' }
        )
    );
    return retBlob;
}

/**
 *  Function that parses JSON data in a web-worker if possible.
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
 *  This function is called to erase some elements from bufferL2 array and avoid
 *  consuming too much memory. 
 */
function freeBuffer(){
    var section = Math.floor(tsVol.currentTimePoint/tsVol.bufferSize);
    var bufferedElements = Object.keys(tsVol.bufferL2).length;
    if(bufferedElements > tsVol.bufferL2Size){
        for(var idx in tsVol.bufferL2){
            if ( idx < (section-tsVol.bufferL2Size/2)%tsVol.timeLength ||
                 idx > (section+tsVol.bufferL2Size/2)%tsVol.timeLength){
                //console.log("erase:", idx)
                delete tsVol.bufferL2[idx];
            }
        }
    }
}

/**
 *  This function is called whenever we can, to load some data ahead of were we're looking.
 */
function streamToBuffer(){
    if(tsVol.requestQueue.length < 2){
        var section = Math.floor(tsVol.currentTimePoint/tsVol.bufferSize);
        var maxSections = Math.floor(tsVol.timeLength/tsVol.bufferSize);
        var from = "from_idx="+tsVol.currentTimePoint;
        var to = ";to_idx=" + Math.min(tsVol.bufferSize + tsVol.currentTimePoint, tsVol.timeLength);
        var query = tsVol.dataAddress+from+to;
        for( var i = 0; i <= tsVol.lookAhead; i++ ){
            var tmp = (section+i)%maxSections;
            if(!tsVol.bufferL2[tmp] && tsVol.requestQueue.indexOf(tmp) < 0) {
                asyncRequest(query, tmp);
                return;
            }
        }
    }
}

/**
 *  A helper function similar to python's range().
 * It takes an integer n and returns an array with all integers from 0 to n-1
 */
function range(len){
    return Array.apply(null, new Array(len)).map(function (_, i) {return i;});
}

/**
 *  This functions returns the X,Y,Z data from time-point t.
 */
function getSliceAtTime(t){
    var buffer;
    var from = "from_idx=" + t;
    var to = ";to_idx=" + Math.min(tsVol.bufferSize + t, tsVol.timeLength);
    var query = tsVol.dataAddress + from + to;
    var section = Math.floor(t/tsVol.bufferSize);
    /*
     Here we erase failed requests from the request queue. 
     TODO: This should happen on a fail handler.
    */
    for(var i in tsVol.requestQueue){
        var currentStep = tsVol.requestQueue[i];
        if(  ( currentStep < section || currentStep > section + tsVol.lookAhead ) &&
             !( currentStep in range(tsVol.lookAhead) )) {
            tsVol.requestQueue.splice(i, 1);
        }
    } 

    if(tsVol.bufferL2[section]){
        buffer = tsVol.bufferL2[section];
    }else{
        buffer = HLPR_readJSONfromFile(query);
        tsVol.bufferL2[section] = buffer;
        tsVol.requestQueue.splice(section, 1);
    }
    return buffer[t%tsVol.bufferSize];
}


// ==================================== DRAWING FUNCTIONS START =============================================

/**
 * Draws the current view depending on the selected entity
 */
// TODO: since only two dimensions change at every time, redraw just those quadrants
// NOTE: this is true only when we navigate, not when we play the time-series
function drawSceneFunctionalFromCube(tIndex) {
    var i, j, k, ii, jj, kk;

    // if we pass no tIndex the function will play
    // from the tsVol.currentTimePoint and increment it
    if(tIndex == null){
        tIndex = tsVol.currentTimePoint;
        tsVol.currentTimePoint++;
        tsVol.currentTimePoint = tsVol.currentTimePoint%tsVol.timeLength;
        tsVol.data = getSliceAtTime(tIndex);
    }
    _setCtxOnQuadrant(0);
    tsVol.ctx.fillStyle = getGradientColorString(tsVol.minimumValue, tsVol.minimumValue, tsVol.maximumValue);
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
    drawNavigator();
    updateMoviePlayerSlider();
}

function drawSceneFunctionalFromView(tIndex) {
    var i, j, k, ii, jj, kk;
    
    // if we pass no tIndex the function will play
    // from the tsVol.currentTimePoint and increment it
    if(tIndex == null){
        tIndex = tsVol.currentTimePoint;
        tsVol.currentTimePoint++;
        tsVol.currentTimePoint = tsVol.currentTimePoint%tsVol.timeLength;
    }
    // query preparation
    var from = "from_idx="+(tIndex);
    var to = ";to_idx="+(tIndex+1);
    var xPlane = ";x_plane="+(tsVol.selectedEntity[0]);
    var yPlane = ";y_plane="+(tsVol.selectedEntity[1]);
    var zPlane = ";z_plane="+(tsVol.selectedEntity[2]);

    var query = tsVol.dataView+from+to+xPlane+yPlane+zPlane;

    // An array containing the view for each plane.
    var sliceArray = HLPR_readJSONfromFile(query);

    _setCtxOnQuadrant(0);
    tsVol.ctx.fillStyle = getGradientColorString(tsVol.minimumValue, tsVol.minimumValue, tsVol.maximumValue);
    tsVol.ctx.fillRect(0, 0, tsVol.ctx.canvas.width, tsVol.ctx.canvas.height);

    for (j = 0; j < tsVol.dataSize[2]; ++j)
        for (i = 0; i < tsVol.dataSize[1]; ++i)
            drawVoxel(i, j, sliceArray[0][0][i][j])
    drawMargin();

    _setCtxOnQuadrant(1);
    for (k = 0; k < tsVol.dataSize[3]; ++k)
        for (jj = 0; jj < tsVol.dataSize[2]; ++jj)
            drawVoxel(k, jj, sliceArray[1][0][jj][k])
    drawMargin();

    _setCtxOnQuadrant(2);
    for (kk = 0; kk < tsVol.dataSize[3]; ++kk)
        for (ii = 0; ii < tsVol.dataSize[1]; ++ii)
            drawVoxel(kk, ii, sliceArray[2][0][ii][kk])
    drawMargin();
    drawNavigator();
    updateMoviePlayerSlider();  
}

function drawSceneFunctional(tIndex){
    if(tsVol.playerIntervalID){
        drawSceneFunctionalFromView(tIndex)
    }
    else{
        drawSceneFunctionalFromCube(tIndex)
    }
}

/**
 * Draws the voxel set at (line, col) in the current quadrant, and colors it according to its value.
 * This function now nothing about the time point. 
 */
function drawVoxel(line, col, value) {
    tsVol.ctx.fillStyle = getGradientColorString(value, tsVol.minimumValue, tsVol.maximumValue);
    // col increases horizontally and line vertically, so col represents the X drawing axis, and line the Y
	tsVol.ctx.fillRect(col * tsVol.currentQuadrant.entityWidth, line * tsVol.currentQuadrant.entityHeight,
	                   tsVol.currentQuadrant.entityWidth+1, tsVol.currentQuadrant.entityHeight+1);
}

/**
 * Draws the cross-hair on each quadrant, on the <code>tsVol.selectedEntity</code>
 */
function drawNavigator() {
    tsVol.ctx.save();
    tsVol.ctx.beginPath();

    for (var quadIdx = 0; quadIdx < 3; ++quadIdx) {
        _setCtxOnQuadrant(quadIdx);
        var x = tsVol.selectedEntity[tsVol.currentQuadrant.axes.x] * tsVol.currentQuadrant.entityWidth + tsVol.currentQuadrant.entityWidth / 2;
        var y = tsVol.selectedEntity[tsVol.currentQuadrant.axes.y] * tsVol.currentQuadrant.entityHeight + tsVol.currentQuadrant.entityHeight / 2;
        drawCrossHair(x, y);
    }
    tsVol.ctx.strokeStyle = "red";
    tsVol.ctx.lineWidth = 3;
    tsVol.ctx.stroke();
    tsVol.ctx.restore();
}

/**
 * Draws a 20px X 20px cross hair on the <code>tsVol.currentQuadrant</code>, at the specified x and y
 */
function drawCrossHair(x, y) {
    tsVol.ctx.moveTo(Math.max(x - 20, 0), y);                              // the horizontal line
    tsVol.ctx.lineTo(Math.min(x + 20, tsVol.quadrantWidth), y);
    tsVol.ctx.moveTo(x, Math.max(y - 20, 0));                              // the vertical line
    tsVol.ctx.lineTo(x, Math.min(y + 20, tsVol.quadrantHeight));
}

/**
 * Draws a 5px rectangle around the <code>tsVol.currentQuadrant</code>
 */
function drawMargin(){
    //_setCtxOnQuadrant(0);
    tsVol.ctx.beginPath();
    tsVol.ctx.rect(2.5, 0, tsVol.quadrantWidth-2, tsVol.quadrantHeight);
    tsVol.ctx.lineWidth = 5;
    tsVol.ctx.strokeStyle = 'black';
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
function _setCtxOnQuadrant(quadIdx) {
    tsVol.currentQuadrant = tsVol.quadrants[quadIdx];
    tsVol.ctx.setTransform(1, 0, 0, 1, 0, 0);                              // reset the transformation
    tsVol.ctx.translate(quadIdx * tsVol.quadrantWidth + tsVol.currentQuadrant.offsetX, tsVol.currentQuadrant.offsetY)
}

/**
 * Returns the number of elements on the given axis
 * @param axis The axis whose length is returned; i=0, j=1, k=2
 * @returns {*}
 * @private
 */
function _getDataSize(axis) {
    switch (axis) {
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
function _getEntityDimensions(xAxis, yAxis) {
    var scaleOnWidth  = tsVol.quadrantWidth  / (_getDataSize(xAxis) * tsVol.voxelSize[xAxis]);
    var scaleOnHeight = tsVol.quadrantHeight / (_getDataSize(yAxis) * tsVol.voxelSize[yAxis]);
    var scale = Math.min(scaleOnHeight, scaleOnWidth);
    return {width: tsVol.voxelSize[xAxis] * scale, height: tsVol.voxelSize[yAxis] * scale}
}

/**
 * Initializes the <code>tsVol.quadrants</code> with some default axes and sets their properties
 */
function _setupQuadrants() {
    tsVol.quadrants.push(new Quadrant({ index: 0, axes: {x: 1, y: 0} }));
    tsVol.quadrants.push(new Quadrant({ index: 1, axes: {x: 1, y: 2} }));
    tsVol.quadrants.push(new Quadrant({ index: 2, axes: {x: 0, y: 2} }));

    for (var quadIdx = 0; quadIdx < tsVol.quadrants.length; quadIdx++) {
        var entityDimensions = _getEntityDimensions(tsVol.quadrants[quadIdx].axes.x, tsVol.quadrants[quadIdx].axes.y);
        tsVol.quadrants[quadIdx].entityHeight = entityDimensions.height;
        tsVol.quadrants[quadIdx].entityWidth  = entityDimensions.width;
        tsVol.quadrants[quadIdx].offsetY = 0;
        tsVol.quadrants[quadIdx].offsetX = 0;
    }
}

/**
 * Values here are usually around 10 or less for <code>bufferSize</code>
 * and less than 200 for <code>bufferL2Size</code>. 
 * TODO: have modifiable limits while suggesting an optimal, safe setup.
 */
function _setupBuffersSize() {
    var tpSize = tsVol.entitySize[0] * tsVol.entitySize[1] * tsVol.entitySize[2];
    //enough to avoid waisting bandwidth and to parse the json smoothly
    while(tsVol.bufferSize * tpSize <= 1000000){
        tsVol.bufferSize++;
    }
    //Very safe measure to avoid crashes. Tested on Chrome.
    while(tsVol.bufferSize * tpSize * tsVol.bufferL2Size <= 157286400){
        tsVol.bufferL2Size *= 2;
    }
    tsVol.bufferL2Size/= 2
}

// ==================================== PRIVATE FUNCTIONS  END  =============================================

// ==================================== PICKING RELATED CODE START ==========================================

function customMouseDown(e) {
    this.mouseDown = true;            // `this` is the canvas
    TSV_pick(e)
}

function customMouseUp(e) {
    this.mouseDown = false;
    if(tsVol.resumePlayer){
        playBack();
        tsVol.resumePlayer = false;
    }
}

function customMouseMove(e) {
    if (!this.mouseDown){
        return;
    }
    TSV_pick(e);
}

/**
 * Implements picking and redraws the scene. Updates sliders too.
 */
function TSV_pick(e){
    if(tsVol.playerIntervalID){
        stopPlayback();
        tsVol.resumePlayer = true;
    }
    //fix for Firefox
    var offset = $('#volumetric-ts-canvas').offset();
    var xpos = e.pageX - offset.left;
    var ypos = e.pageY - offset.top;
    var selectedQuad = tsVol.quadrants[Math.floor(xpos / tsVol.quadrantWidth)];
    // check if it's inside the quadrant but outside the drawing
    if (ypos < selectedQuad.offsetY || ypos >= tsVol.quadrantHeight - selectedQuad.offsetY ||
        xpos < tsVol.quadrantWidth * selectedQuad.index + selectedQuad.offsetX ||
        xpos >= tsVol.quadrantWidth * (selectedQuad.index + 1) - selectedQuad.offsetX)
        return;
    var selectedEntityOnX = Math.floor((xpos % tsVol.quadrantWidth) / selectedQuad.entityWidth);
    var selectedEntityOnY = Math.floor((ypos - selectedQuad.offsetY) / selectedQuad.entityHeight);
    tsVol.selectedEntity[selectedQuad.axes.x] = selectedEntityOnX;
    tsVol.selectedEntity[selectedQuad.axes.y] = selectedEntityOnY;
    updateSliders();
    drawSceneFunctional(tsVol.currentTimePoint);
}

// ==================================== PICKING RELATED CODE  END  ==========================================

// ==================================== UI RELATED CODE START ===============================================
/**
 * Functions that calls all the setup functions for the main UI of the time series volume visualizer.
*/
function TSV_startUserInterface(){
    startButtonSet();
    startPositionSliders();
    startMovieSlider();
}

/**
 * Function that creates all the buttons for playing, stopping and seeking time points.
*/
function startButtonSet(){
	// we setup every button
    var container = $('#buttons');
    var first = $('<button id="first">').button({icons:{primary:"ui-icon-seek-first"}});
    var prev = $('<button id="prev">').button({icons:{primary:"ui-icon-seek-prev"}});
    var playButton = $('<button id="play">').button({icons:{primary:"ui-icon-play"}});
    var stopButton = $('<button id="stop">').button({icons:{primary:"ui-icon-stop"}});
    var next = $('<button id="next">').button({icons:{primary:"ui-icon-seek-next"}});
    var end = $('<button id="end">').button({icons:{primary:"ui-icon-seek-end"}});

    // put the buttons on an array for easier manipulation
    var buttonsArray = [first, prev, playButton, stopButton, next, end];

    //we attach event listeners to buttons as needed
    playButton.click(playBack);
    stopButton.click(stopPlayback);
    prev.click(playPreviousTimePoint);
    next.click(playNextTimePoint);
    first.click(seekFirst);
    end.click(seekEnd);

    // we setup the DOM element that will contain the buttons
    container.buttonset();

    // add every button to the container and refresh it afterwards
    buttonsArray.forEach(function(entry){
        container.append(entry)
    });
    container.buttonset('refresh');
}

/**
 * Code for the navigation slider. Creates the x,y,z sliders and adds labels
*/
function startPositionSliders(){
    var i = 0;
    var axArray = ["X", "Y", "Z"];
    // We loop trough every slider
    $( "#sliders").find("> span" ).each(function(){
        var value = tsVol.selectedEntity[i];
        var opts = {
                        value: value,
                        min: 0,
                        max: tsVol.entitySize[i]-1, // yeah.. if we start from zero we need to subtract 1
                        animate: true,
                        orientation: "horizontal",
                        change: slideMove, // call this function *after* the slide is moved OR the value changes
                        slide: slideMove  //  we use this to keep it smooth.   
                    };
        $(this).slider(opts).each(function(){
            // The starting time point. Supposing it is always ZERO.
            var el = $('<label class="min-coord">'+opts.min+'</label>');
            $(this).append(el);
            // The name of the slider
            el = $('<label class="axis-name">'+axArray[i]+' Axis'+'</label>');        
            $(this).append(el);
            // The current value of the slider
            el = $('<label class="slider-value" id="slider-'+axArray[i]+'-value">['+value+']</label>');    
            $(this).append(el);
            // The maximum value for the slider
            el = $('<label class="max-coord">'+opts.max+'</label>');
            $(this).append(el);
            i++; // let's do the same on the next slider now..
        })
    });
}

/**
 * Code for "movie player" slider. Creates the slider and adds labels
*/
function startMovieSlider(){
	    $("#time-position").find("> span").each(function(){
        var value = 0;
        var opts = {
                    value: value,
                    min: 0,
                    max: tsVol.timeLength,
                    animate: true,
                    orientation: "horizontal",
                    range: "min",
                    stop: moviePlayerMoveEnd,
                    slide: moviePlayerMove
                    };
        $(this).slider(opts).each(function(){
            // The starting point. Supposing it is always ZERO.
            var el = $('<label id="time-slider-min">'+opts.min+'</label>');
            $(this).append(el);
            // The actual time point we are seeing
            el = $('<label id="time-slider-value">'+value+'/'+tsVol.timeLength+'</label>');
            $(this).append(el);
        });
    });
}

// ==================================== CALLBACK FUNCTIONS START ===============================================

function playBack(){
    if(!tsVol.playerIntervalID)
        tsVol.playerIntervalID = window.setInterval(drawSceneFunctional, tsVol.playbackRate);
}

function stopPlayback(){
    window.clearInterval(tsVol.playerIntervalID);
    tsVol.playerIntervalID = null;
}

function playNextTimePoint(){
    tsVol.currentTimePoint++;
    tsVol.currentTimePoint = tsVol.currentTimePoint%(tsVol.timeLength+1);
    drawSceneFunctional(tsVol.currentTimePoint)
}

function playPreviousTimePoint(){
    if(tsVol.currentTimePoint === 0)
        tsVol.currentTimePoint = tsVol.timeLength+1;
    drawSceneFunctional(--tsVol.currentTimePoint)
}

function seekFirst(){
    tsVol.currentTimePoint = 0;
    drawSceneFunctional(tsVol.currentTimePoint);
}

function seekEnd(){
    tsVol.currentTimePoint = tsVol.timeLength;
    drawSceneFunctional(tsVol.currentTimePoint-1);
}

// Updates the position and values of the x,y,z navigation sliders when we click the canvas.
function updateSliders(){
    var axArray = ["X", "Y", "Z"];
    var i = 0;
    $( "#sliders").find("> span" ).each(function(){
        $(this).slider("option", "value", tsVol.selectedEntity[i]); //update the handle
        $('slider-'+axArray[i]+'-value').empty().text( '['+tsVol.selectedEntity[i]+']' ); //update the label
        i += 1;
    });
}

// Updated the player slider bar while playback is on.
function updateMoviePlayerSlider(){
    $("#time-position").find("> span").each(function(){
        $(this).slider("option", "value", tsVol.currentTimePoint);
        $('#time-slider-value').empty().text( tsVol.currentTimePoint+'/'+tsVol.timeLength );
    })
}

// When the navigation sliders are moved, this redraws the scene accordingly.
function slideMove(event, ui){
    var quadID = ["x-slider", "y-slider", "z-slider"].indexOf(event.target.id);
    var selectedQuad = tsVol.quadrants[quadID];

    //  Updates the label value on the slider.
    $(event.target.children[3]).empty().text( '['+ui.value+']' );
    //  Setup things to draw the scene pointing to the right voxel and redraw it.
    if(quadID == 1)
        tsVol.selectedEntity[selectedQuad.axes.x] = ui.value;
    else
        tsVol.selectedEntity[selectedQuad.axes.y] = ui.value;
    drawSceneFunctional(tsVol.currentTimePoint);
}

// Updates the value at the end of the player bar when we move the handle.
function moviePlayerMove(event, ui){
    $("#time-position").find("> span").each(function(){
        $('#time-slider-value').empty().text( ui.value+'/'+tsVol.timeLength );
    })
}

/*
* Redraws the scene at the selected time-point at the end of a slide action.
* Calling this during the the whole slide showed to be too expensive.
* Thus, the new time-point is drawn only when the user releases the click from the handler
*/
function moviePlayerMoveEnd(event, ui){
    tsVol.currentTimePoint = ui.value;
    drawSceneFunctional(tsVol.currentTimePoint);
}

// ==================================== CALLBACK FUNCTIONS END ===============================================
// ==================================== UI RELATED CODE END =================================================