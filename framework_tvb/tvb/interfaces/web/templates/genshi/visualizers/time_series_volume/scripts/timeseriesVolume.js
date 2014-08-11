
var tsVol = { 
    ctx: null,                  // the context for drawing on current canvas.
    currentQuadrant: 0,         // the quadrant we're in.
    quadrants: [],              // the quadrants array.
    minimumValue: null,         // minimum value of the dataset.
    maximumValue: null,         // maximum value of the dataset.
    voxelSize: null,
    volumeOrigin: null,         // volumeOrigin is not used for now. if needed, use it in _setQuadrant
    selectedEntity: [0, 0, 0],  // the selected voxel; [i, j, k].
    selectedEntityValue: 0,
    entitySize: [0, 0, 0],      // the size of each plane
    quadrantHeight: null,       // the height of the three small left quadrants
    quadrantWidth: null,        // the width of the three small left quadrants
    focusQuadrantHeight: null,  // the height of the focus quadrant
    focusQuadrantWidth: null,   // the width of the focus quadrant
    legendHeight: 0,            // the height of the legend quadrant
    legendWidth: 0,             // the width of the legend quadrant
    legendQuadrant: null,       // the object tat will contain the legend quadrant
    legendPadding:80*2,
    selectedQuad: 0,            // the quadrant selected by the user every time
    highlightedQuad: {},        // the plane to be displayed on the focus quadrant
    timeLength: 0,              // number of timepoints in the Volume.
    currentTimePoint: 0,        
    playbackRate: 66,           // This is a not acurate lower limit for playback speed.
    playerIntervalID: null,     // ID from the player's setInterval().
    streamToBufferID: null,     // ID from the buffering system's setInterval().
    bufferSize: 1,              // How many time points do we load each time?
    bufferL2Size: 1,            // How many sets of buffers can we keep at the same time?
    lookAhead: 100,             // How many sets of buffers should be loaded ahead of us each time?
    data: {},                   // The actual data to be drawn to canvas.
    sliceArray: [],             // A helper variable to draw the data on the canvas.
    bufferL2: {},               // Cotains all data loaded and preloaded, limited by memory.
    bufferL3: {},               // Cotains all data from loaded views, limited by memory.
    dataAddress: "",            // Used to contain the python URL for each time point.
    dataView: "",               // Used to store the call for get_volume_view server function.
    dataSize: "",               // Used first to contain the file ID and then it's dimension.
    requestQueue: [],           // Used to avoid requesting a time point set while we are waiting for it.
    parserBlob: null,           // Used to store the JSON Parser Blob for web-workers.
    slidersClicked: false,      // Used to handle the status of the sliders.
    batchID: 0,                 // Used to ignore useless incoming ajax responses.
    dataTimeSeries: "",         // Contains the address to query the time series of a voxel.
    tsDataArray: [],
    samplePeriod: 0,
    samplePeriodUnit: "",
    relevantFeature: "mean"
};

var Quadrant = function (params){                // this keeps all necessary data for drawing
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
function TSV_initVisualizer(dataUrls, minValue, maxValue, volOrigin, sizeOfVoxel){
    var canvas = document.getElementById("volumetric-ts-canvas");
    if (!canvas.getContext){
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

    // var canvasWidth = $(canvas).parent().width();
    // canvas.width  = canvasWidth;                          // fill the screen on width
    // canvas.height = canvasWidth / 3;                      // three quadrants

    canvas.height = $(canvas).parent().height();
    canvas.width  = $(canvas).parent().width();

    tsVol.quadrantHeight = canvas.height / 3;               // quadrants are squares
    tsVol.quadrantWidth = tsVol.quadrantHeight;
    tsVol.focusQuadrantWidth = canvas.width - tsVol.quadrantWidth;
    tsVol.legendHeight = canvas.height / 13;
    tsVol.legendWidth = tsVol.focusQuadrantWidth - tsVol.legendPadding;
    tsVol.focusQuadrantHeight = canvas.height - tsVol.legendHeight;

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
    tsVol.dataTimeSeries = dataUrls[3];
    
    var tmp = HLPR_readJSONfromFile(dataUrls[4]);
    tsVol.samplePeriod = tmp[0];
    tsVol.samplePeriodUnit = tmp[1];

    tsVol.minimumValue = minValue;
    tsVol.maximumValue = maxValue;
    tsVol.timeLength = tsVol.dataSize[0]; //Number of time points;

    _setupQuadrants();

    tsVol.selectedEntity[0] = Math.floor(tsVol.dataSize[1] / 2); // set the center entity as the selected one
    tsVol.selectedEntity[1] = Math.floor(tsVol.dataSize[2] / 2);
    tsVol.selectedEntity[2] = Math.floor(tsVol.dataSize[3] / 2);

    tsVol.entitySize[0] = tsVol.dataSize[1];    // get entities number of voxels
    tsVol.entitySize[1] = tsVol.dataSize[2];
    tsVol.entitySize[2] = tsVol.dataSize[3];
    tsVol.entitySize[3] = tsVol.dataSize[0];    // get entities number of time points

    _setupBuffersSize();

    ColSch_initColorSchemeParams(minValue, maxValue, colorRedraw);
    tsVol.currentTimePoint = 0;
    //tsVol.highlightedQuad.index = 0;

    startBuffering();
    window.setInterval(freeBuffer, tsVol.playbackRate*10);
    drawGraphs();
}

/**
 * Requests file data not blocking the main thread if possible.
 */
function asyncRequest(fileName, sect){
    var index = tsVol.requestQueue.indexOf(sect);
    var privateID = tsVol.batchID;
    if (index < 0){
        tsVol.requestQueue.push(sect);
        doAjaxCall({
            async:true,
            url:fileName,
            methos:"GET",
            mimeType:"text/plain",
            success:function(r){
                if(privateID == tsVol.batchID){
                    parseAsync(r, function(json){
                        tsVol.bufferL3[sect] = json;
                        idx = tsVol.requestQueue.indexOf(sect);
                        if (idx > -1){
                            tsVol.requestQueue.splice(idx, 1);
                        }   
                    });
                }
            }
        });
    }
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
 *  This function is called to erase some elements from bufferL3 array and avoid
 *  consuming too much memory. 
 */
function freeBuffer(){
    var section = Math.floor(tsVol.currentTimePoint/tsVol.bufferSize);
    var bufferedElements = Object.keys(tsVol.bufferL3).length;
    if(bufferedElements > tsVol.bufferL2Size){
        tsVol.bufferL2 = {};
        for(var idx in tsVol.bufferL3){
            if ( idx < (section - tsVol.bufferL2Size/2) % tsVol.timeLength ||
                 idx > (section + tsVol.bufferL2Size/2) % tsVol.timeLength){
                delete tsVol.bufferL3[idx];
            }
        }
    }
}

/**
 *  This function is called whenever we can, to load some data ahead of were we're looking.
 */
function streamToBuffer(){
    if(tsVol.requestQueue.length < 2){ //avoid having too many requests at the same time
        var currentSection = Math.floor(tsVol.currentTimePoint/tsVol.bufferSize);
        var maxSections = Math.floor(tsVol.timeLength/tsVol.bufferSize);
        var xPlane = ";x_plane=" + (tsVol.selectedEntity[0]);
        var yPlane = ";y_plane=" + (tsVol.selectedEntity[1]);
        var zPlane = ";z_plane=" + (tsVol.selectedEntity[2]);
        
        for( var i = 0; i <= tsVol.lookAhead; i++ ){
            var toBufferSection = Math.min( currentSection + i, maxSections );
            if(!tsVol.bufferL3[toBufferSection] && tsVol.requestQueue.indexOf(toBufferSection) < 0){
                var from = toBufferSection*tsVol.bufferSize;
                var to = from+tsVol.bufferSize;
                from = "from_idx=" + from;
                to = ";to_idx=" + to;
                var query = tsVol.dataView + from + to + xPlane + yPlane + zPlane;
                asyncRequest(query, toBufferSection);
                return;
            }
        }
    }
}

/**
 * A helper function similar to python's range().
 * It takes an integer n and returns an array with all integers from 0 to n-1
 */
function range(len){
    return Array.apply(null, new Array(len)).map(function (_, i){return i;});
}

/**
 *  This functions returns the whole x,y,z cube data at time-point t.
 */
function getSliceAtTime(t){
    var buffer;
    var from = "from_idx=" + t;
    var to = ";to_idx=" + (t +1);
    var query = tsVol.dataAddress + from + to; 

    if(tsVol.bufferL2[t]){
        buffer = tsVol.bufferL2[t];
    }else{
        //window.setTimeout(function(){showBlockerOverlay(2000);},0);
        tsVol.bufferL2[t] = HLPR_readJSONfromFile(query);
        //window.setTimeout(function(){closeBlockerOverlay();},0);
        buffer = tsVol.bufferL2[t];
    }
    return buffer[0];
}

/**
 *  This functions returns the X,Y,Z data from time-point t.
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

    if(tsVol.bufferL3[section]){ //We have that slice in memory
        buffer = tsVol.bufferL3[section];
    }else{ // We need to load that slice from
        from = "from_idx=" + t;
        to = ";to_idx=" + Math.min(1 + t, tsVol.timeLength);
        query = tsVol.dataView + from + to + xPlane + yPlane + zPlane;

        buffer = HLPR_readJSONfromFile(query);
        return [buffer[0][0],buffer[1][0],buffer[2][0]];
    }
    t = t%tsVol.bufferSize;
    return [buffer[0][t],buffer[1][t],buffer[2][t]];
}


// ==================================== DRAWING FUNCTIONS START =============================================

/**
 * Draws the current view depending on the selected entity
 */
function drawSceneFunctional(tIndex){
    if(tsVol.playerIntervalID){
        drawSceneFunctionalFromView(tIndex)
    }
    else{
        drawSceneFunctionalFromCube(tIndex)
    }
    drawLegend();
}
/**
 * Necessary for the color picking function
 */
function colorRedraw(){
    drawSceneFunctional(tsVol.currentTimePoint);
}

/**
 * Draws the current scene from the whole loaded cube data
 */
function drawSceneFunctionalFromCube(tIndex){
    var i, j, k, ii, jj, kk;

    // if we pass no tIndex the function will play
    // from the tsVol.currentTimePoint and increment it
    if(tIndex == null){
        tIndex = tsVol.currentTimePoint;
        tsVol.currentTimePoint++;
        tsVol.currentTimePoint = tsVol.currentTimePoint % tsVol.timeLength;
    }

    tsVol.data = getSliceAtTime(tIndex);
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
    drawFocusQuadrantFromCube(tIndex);
    drawNavigator();
    updateMoviePlayerSlider();
    setSelectedEntityValue(tsVol.data);
}

function setSelectedEntityValue(data){
    if(data.length == 3){
        tsVol.selectedEntityValue = data[0][tsVol.selectedEntity[0]][tsVol.selectedEntity[1]];
    }
    else{
        tsVol.selectedEntityValue = data[tsVol.selectedEntity[0]][tsVol.selectedEntity[1]][tsVol.selectedEntity[2]];
    }
}

function drawFocusQuadrantFromCube(tIndex){
    _setCtxOnQuadrant(3);
    if(tsVol.highlightedQuad.index == 0){
        for (j = 0; j < tsVol.dataSize[2]; ++j)
            for (i = 0; i < tsVol.dataSize[1]; ++i)
                drawVoxel(i, j, tsVol.data[i][j][tsVol.selectedEntity[2]]);
    }
    else if(tsVol.highlightedQuad.index == 1){
        for (k = 0; k < tsVol.dataSize[3]; ++k)
            for (jj = 0; jj < tsVol.dataSize[2]; ++jj)
                drawVoxel(k, jj, tsVol.data[tsVol.selectedEntity[0]][jj][k]);
    }
    else if(tsVol.highlightedQuad.index == 2){
        for (kk = 0; kk < tsVol.dataSize[3]; ++kk)
            for (ii = 0; ii < tsVol.dataSize[1]; ++ii)
                drawVoxel(kk, ii, tsVol.data[ii][tsVol.selectedEntity[1]][kk]);
    }
    drawMargin();   
}

/**
 * Draws the current scene only from the three visible planes data.
 */
function drawSceneFunctionalFromView(tIndex){
    var i, j, k, ii, jj, kk;
    
    // if we pass no tIndex the function will play
    // from the tsVol.currentTimePoint and increment it
    if(tIndex == null){
        tIndex = tsVol.currentTimePoint;
        tsVol.currentTimePoint++;
        tsVol.currentTimePoint = tsVol.currentTimePoint % tsVol.timeLength;
    }
    // An array containing the view for each plane.
    tsVol.sliceArray = getViewAtTime(tIndex);

    _setCtxOnQuadrant(0);
    tsVol.ctx.fillStyle = getGradientColorString(tsVol.minimumValue, tsVol.minimumValue, tsVol.maximumValue);
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

function drawFocusQuadrantFromView(tIndex){
    _setCtxOnQuadrant(3);
    if(tsVol.highlightedQuad.index == 0){
        for (j = 0; j < tsVol.dataSize[2]; ++j)
            for (i = 0; i < tsVol.dataSize[1]; ++i)
                drawVoxel(i, j, tsVol.sliceArray[0][i][j]);
    }
    else if(tsVol.highlightedQuad.index == 1){
        for (k = 0; k < tsVol.dataSize[3]; ++k)
            for (jj = 0; jj < tsVol.dataSize[2]; ++jj)
                drawVoxel(k, jj, tsVol.sliceArray[1][jj][k]);
    }
    else if(tsVol.highlightedQuad.index == 2){
        for (kk = 0; kk < tsVol.dataSize[3]; ++kk)
            for (ii = 0; ii < tsVol.dataSize[1]; ++ii)
                drawVoxel(kk, ii, tsVol.sliceArray[2][ii][kk]);
    }
    drawMargin();   
}

/**
 * Draws the voxel set at (line, col) in the current quadrant, and colors it according to its value.
 * This function now nothing about the time point. 
 */
function drawVoxel(line, col, value){
    tsVol.ctx.fillStyle = getGradientColorString(value, tsVol.minimumValue, tsVol.maximumValue);
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
    var x = tsVol.selectedEntity[tsVol.currentQuadrant.axes.x] * tsVol.currentQuadrant.entityWidth + tsVol.currentQuadrant.entityWidth / 2;
    var y = tsVol.selectedEntity[tsVol.currentQuadrant.axes.y] * tsVol.currentQuadrant.entityHeight + tsVol.currentQuadrant.entityHeight / 2;
    drawFocusCrossHair(x, y);

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
function drawLegend(){
    var tmp = tsVol.legendPadding / 2;
    // set the context on the legend quadrant
    tsVol.ctx.setTransform(1, 0, 0, 1, 0, 0);  // reset the transformation
    tsVol.ctx.translate( tsVol.quadrantWidth + tmp, tsVol.focusQuadrantHeight);
    tsVol.ctx.font = 'italic 18px Arial';
    tsVol.ctx.textAlign = 'center';
    tsVol.ctx. textBaseline = 'middle';
    tsVol.ctx.fillStyle = 'white';  // a color name or by using rgb/rgba/hex values
    tsVol.ctx.fillText(nFormatter(tsVol.minimumValue), 0, tsVol.legendHeight - 10); // text and position
    tsVol.ctx.fillText("|", 1, tsVol.legendHeight/2);
    var midValue = (tsVol.maximumValue - tsVol.minimumValue)/2;
    tsVol.ctx.fillText(nFormatter(midValue), tsVol.legendWidth/2, tsVol.legendHeight - 10); // text and position
    tsVol.ctx.fillText("|", tsVol.legendWidth/2, tsVol.legendHeight/2);
    tsVol.ctx.fillText(nFormatter(tsVol.maximumValue), tsVol.legendWidth, tsVol.legendHeight - 10); // text and position
    tsVol.ctx.fillText("|", tsVol.legendWidth-1, tsVol.legendHeight/2);

    for(var i = 0; i< tsVol.legendWidth; i++){
        var val = tsVol.minimumValue + ((i/tsVol.legendWidth)*(tsVol.maximumValue-tsVol.minimumValue));
        tsVol.ctx.fillStyle = getGradientColorString(val, tsVol.minimumValue, tsVol.maximumValue);
        tsVol.ctx.fillRect(i, 1, 1.5, tsVol.legendHeight/2);
    }
    tsVol.ctx.fillStyle = "white";
    tmp = tsVol.selectedEntityValue;
    tmp = tmp / (tsVol.maximumValue-tsVol.minimumValue);
    tmp = (tmp*tsVol.legendWidth);
    tsVol.ctx.fillRect(tmp, 1, 3, tsVol.legendHeight/2);
}
/**
 * Formats numbers in the 'K', 'M', 'G' notation, e.g. 2300 becomes 2.3K.
 */
function nFormatter(num){
     if (num >= 1000000000){
        return (num / 1000000000).toFixed(1).replace(/\.0$/, '') + 'G';
     }
     if (num >= 1000000){
        return (num / 1000000).toFixed(1).replace(/\.0$/, '') + 'M';
     }
     if (num >= 1000){
        return (num / 1000).toFixed(1).replace(/\.0$/, '') + 'K';
     }
     return num;
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
    //tsVol.ctx.rect(2.5, 0, tsVol.quadrantWidth-2, tsVol.quadrantHeight);
    tsVol.ctx.rect(2.5, 0, marginWidth-2, marginHeight-2);
    tsVol.ctx.lineWidth = 5;
    if(tsVol.currentQuadrant.index == tsVol.selectedQuad.index && tsVol.currentQuadrant.index != 3){
        tsVol.ctx.strokeStyle = 'white';
        tsVol.highlightedQuad = tsVol.currentQuadrant;
    }
    else if(tsVol.currentQuadrant.index == tsVol.highlightedQuad.index && tsVol.selectedQuad.index == 3){
        tsVol.ctx.strokeStyle = 'white';
    }
    else{
        tsVol.ctx.strokeStyle = 'black';
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
       tsVol.ctx.translate(1 * tsVol.quadrantWidth + tsVol.currentQuadrant.offsetX, tsVol.currentQuadrant.offsetY); 
    }
    else{
        tsVol.ctx.translate( tsVol.currentQuadrant.offsetX, quadIdx * tsVol.quadrantHeight +  tsVol.currentQuadrant.offsetY);
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
    _setupLegendQuadrant();
}

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

function _setupLegendQuadrant(){
    tsVol.legendQuadrant = new Quadrant({index:4});
    tsVol.legendQuadrant.entityHeight = tsVol.legendHeight;
    tsVol.legendQuadrant.entityWidth  = tsVol.legendWidth;
    tsVol.legendQuadrant.offsetY = tsVol.focusQuadrantHeight;
    tsVol.legendQuadrant.offsetX = tsVol.focusQuadrantWidth;
}

/** 
 * Automathically determine optimal bufferSizer, depending on data dimensions.
 */
function _setupBuffersSize(){
    var tpSize = Math.max(tsVol.entitySize[0], tsVol.entitySize[1], tsVol.entitySize[2]);
    var tpSize = tpSize * tpSize;
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
 */
function TSV_pick(e){
    tsVol.bufferL3 = {};
    stopBuffering();
    if(tsVol.playerIntervalID){
        stopPlayback();
        tsVol.resumePlayer = true;
    }
    //fix for Firefox
    var offset = $('#volumetric-ts-canvas').offset();
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
    }
    else{
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

    //var selectedEntityOnX = Math.floor((xpos % tsVol.quadrantWidth) / selectedQuad.entityWidth);
    //var selectedEntityOnY = Math.floor((ypos - selectedQuad.offsetY) / selectedQuad.entityHeight);
    if(tsVol.selectedQuad.index == 3){
        var selectedEntityOnX = Math.floor(((xpos - tsVol.quadrantWidth) % tsVol.focusQuadrantWidth) / tsVol.selectedQuad.entityWidth);
        //var selectedEntityOnY = Math.floor((ypos - tsVol.selectedQuad.offsetY) / tsVol.selectedQuad.entityHeight);
        var selectedEntityOnY = Math.floor(((ypos - tsVol.selectedQuad.offsetY) % tsVol.focusQuadrantHeight) / tsVol.selectedQuad.entityHeight);
    }
    else{
        var selectedEntityOnX = Math.floor((xpos - tsVol.selectedQuad.offsetX) / tsVol.selectedQuad.entityWidth);
        var selectedEntityOnY = Math.floor((ypos % tsVol.quadrantHeight) / tsVol.selectedQuad.entityHeight);
    }
    tsVol.selectedEntity[tsVol.selectedQuad.axes.x] = selectedEntityOnX;
    tsVol.selectedEntity[tsVol.selectedQuad.axes.y] = selectedEntityOnY;
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
                        change: slideMoved, // call this function *after* the slide is moved OR the value changes
                        slide: slideMove  //  we use this to keep it smooth.   
                    };
        $(this).slider(opts).each(function(){
            // The starting time point. Supposing it is always ZERO.
            var el = $('<label class="min-coord">' + opts.min + '</label>');
            $(this).append(el);
            // The name of the slider
            el = $('<label class="axis-name">' + axArray[i] + ' Axis' + '</label>');
            $(this).append(el);
            // The current value of the slider
            el = $('<label class="slider-value" id="slider-' + axArray[i] + '-value">[' + value + ']</label>');
            $(this).append(el);
            // The maximum value for the slider
            el = $('<label class="max-coord">' + opts.max + '</label>');
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
                    max: tsVol.timeLength - 1,
                    animate: true,
                    orientation: "horizontal",
                    range: "min",
                    stop: moviePlayerMoveEnd,
                    slide: moviePlayerMove
                    };
        $(this).slider(opts).each(function(){
            // The starting point. Supposing it is always ZERO.
            var el = $('<label id="time-slider-min">' + opts.min + '</label>');
            $(this).append(el);
            // The actual time point we are seeing
            var actualTime = value * tsVol.samplePeriod;
            var totalTime = (tsVol.timeLength - 1) * tsVol.samplePeriod;
            el = $('<label id="time-slider-value">' + actualTime.toFixed(2) + '/' + totalTime.toFixed(2) + " (in "+ tsVol.samplePeriodUnit +')</label>');
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
    drawSceneFunctionalFromView(tsVol.currentTimePoint)
}

function playPreviousTimePoint(){
    if(tsVol.currentTimePoint === 0)
        tsVol.currentTimePoint = tsVol.timeLength + 1;
    drawSceneFunctionalFromView(--tsVol.currentTimePoint)
}

function seekFirst(){
    tsVol.currentTimePoint = 0;
    drawSceneFunctionalFromView(tsVol.currentTimePoint);
}

function seekEnd(){
    tsVol.currentTimePoint = tsVol.timeLength - 1;
    drawSceneFunctionalFromView(tsVol.currentTimePoint - 1);
}

// Updates the position and values of the x,y,z navigation sliders when we click the canvas.
function updateSliders(){
    var axArray = ["X", "Y", "Z"];
    var i = 0;
    $( "#sliders").find("> span" ).each(function(){
        $(this).slider("option", "value", tsVol.selectedEntity[i]); //update the handle
        $('slider-' + axArray[i] + '-value').empty().text( '[' + tsVol.selectedEntity[i] + ']' ); //update the label
        i += 1;
    });
}

// Updated the player slider bar while playback is on.
function updateMoviePlayerSlider(){
    $("#time-position").find("> span").each(function(){
        $(this).slider("option", "value", tsVol.currentTimePoint);
        var actualTime = tsVol.currentTimePoint * tsVol.samplePeriod;
        var totalTime = (tsVol.timeLength - 1) * tsVol.samplePeriod;
        $('#time-slider-value').empty().text( actualTime.toFixed(2) + '/' + totalTime.toFixed(2) + (" (in "+ tsVol.samplePeriodUnit +")") );
    });
    d3.select(".timeVerticalLine").attr("transform", function(){
                var width = $(".graph-timeSeries-rect").attr("width");
                var pos = (tsVol.currentTimePoint*width)/(tsVol.timeLength);
                return "translate(" + pos + ", 0)";
            });
}

// While the navigation sliders are moved, this redraws the scene accordingly.
function slideMove(event, ui){
    tsVol.bufferL3 = {};
    stopBuffering();
    if(tsVol.playerIntervalID){
        stopPlayback();
        tsVol.resumePlayer = true;
    }
    tsVol.slidersClicked = true;
    var quadID = ["x-slider", "y-slider", "z-slider"].indexOf(event.target.id);
    var selectedQuad = tsVol.quadrants[quadID];

    //  Updates the label value on the slider.
    $(event.target.children[3]).empty().text( '[' + ui.value + ']' );
    //  Setup things to draw the scene pointing to the right voxel and redraw it.
    if(quadID == 1)
        tsVol.selectedEntity[selectedQuad.axes.x] = ui.value;
    else
        tsVol.selectedEntity[selectedQuad.axes.y] = ui.value;
    drawSceneFunctional(tsVol.currentTimePoint);
}

// After the navigation sliders are changed, this redraws the scene accordingly.
function slideMoved(event, ui){
    if(tsVol.slidersClicked){
        startBuffering();
        tsVol.slidersClicked = false;
    
        if(tsVol.resumePlayer){
            window.setTimeout(playBack, tsVol.playbackRate * 5);
            tsVol.resumePlayer = false;
        }
    }
    var quadID = ["x-slider", "y-slider", "z-slider"].indexOf(event.target.id);
    var selectedQuad = tsVol.quadrants[quadID];

    //  Updates the label value on the slider.
    $(event.target.children[3]).empty().text( '[' + ui.value + ']' );
    //  Setup things to draw the scene pointing to the right voxel and redraw it.
    if(quadID == 1)
        tsVol.selectedEntity[selectedQuad.axes.x] = ui.value;
    else
        tsVol.selectedEntity[selectedQuad.axes.y] = ui.value;
}

// Updates the value at the end of the player bar when we move the handle.
function moviePlayerMove(event, ui){
    $("#time-position").find("> span").each(function(){
        var actualTime = ui.value * tsVol.samplePeriod;
        var totalTime = (tsVol.timeLength - 1) * tsVol.samplePeriod;
        $('#time-slider-value').empty().text( actualTime.toFixed(2) + '/' + totalTime.toFixed(2) + (" (in "+ tsVol.samplePeriodUnit +")") );
    });
        d3.select(".timeVerticalLine").attr("transform", function(){
                var width = $(".graph-timeSeries-rect").attr("width");
                var pos = (ui.value*width)/(tsVol.timeLength);
                return "translate(" + pos + ", 0)";
            });
}

/*
* Redraws the scene at the selected time-point at the end of a slide action.
* Calling this during the whole slide showed to be too expensive.
* Thus, the new time-point is drawn only when the user releases the click from the handler
*/
function moviePlayerMoveEnd(event, ui){
    tsVol.currentTimePoint = ui.value;
    drawSceneFunctionalFromView(tsVol.currentTimePoint);
    drawLegend();
}

// ==================================== CALLBACK FUNCTIONS END ===============================================
// ==================================== UI RELATED CODE END ==================================================
