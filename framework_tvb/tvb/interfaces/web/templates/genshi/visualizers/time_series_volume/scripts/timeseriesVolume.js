/* globals displayMessage, ColSch_initColorSchemeGUI, ColSch_getAbsoluteGradientColorString,
            doAjaxCall, d3, HLPR_readJSONfromFile,
            TSV_initVolumeView, TSV_drawVolumeScene, TSV_hitTest
             */

// ==================================== INITIALIZATION CODE START ===========================================
var tsVol = {
    minimumValue: null,         // Minimum value of the dataset.
    maximumValue: null,         // Maximum value of the dataset.
    volumeOrigin: null,         // VolumeOrigin is not used for now. if needed, use it in _setQuadrant
    selectedEntity: [0, 0, 0],  // The selected voxel; [i, j, k].
    entitySize: [0, 0, 0],      // The size of each plane
    selectedQuad: 0,            // The quadrant selected by the user every time
    timeLength: 0,              // Number of timepoints in the Volume.
    currentTimePoint: 0,
    playbackRate: 66,           // This is a not acurate lower limit for playback speed.
    playerIntervalID: null,     // ID from the player's setInterval().
    streamToBufferID: null,     // ID from the buffering system's setInterval().
    bufferSize: 1,              // How many time points to get each time. It will be computed automatically, This is only the start value
    bufferL2Size: 1,            // How many sets of buffers can we keep at the same time in memory
    lookAhead: 10,             // How many sets of buffers should be loaded ahead of us each time?
    data: {},                   // The actual data to be drawn to canvas.
    bufferL2: {},               // Contains all data from loaded views, limited by memory.
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

var SLIDERS = ["X", "Y", "Z"];
var SLIDERIDS = ["sliderForAxisX", "sliderForAxisY", "sliderForAxisZ"];

/**
 * Make all the necessary initialisations and draws the default view, with the center voxel selected
 * @param urlVolumeData          Url base for retrieving current slices data (for the left-side)
 * @param urlTimeSeriesData      URL base for retrieving TS (right side)
 * @param minValue          The minimum value for all the slices
 * @param maxValue          The maximum value for all the slices
 * @param samplePeriod      Number representing how frequent the signal was being measured
 * @param samplePeriodUnit  Unit-Measure for the sampling period (e.g. ms, s)
 * @param volumeShape       Json with the shape of the full TS
 * @param volOrigin         The origin of the rendering; irrelevant in 2D, for now
 * @param sizeOfVoxel       How the voxel is sized on each axis; [xScale, yScale, zScale]
 */
function TSV_initVisualizer(urlVolumeData, urlTimeSeriesData, minValue, maxValue, samplePeriod, samplePeriodUnit,
                            volumeShape, volOrigin, sizeOfVoxel) {
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
    tsVol.dataSize = $.parseJSON(volumeShape);

    TSV_initVolumeView(tsVol.dataSize, minValue, maxValue, $.parseJSON(sizeOfVoxel));

    tsVol.urlVolumeData = urlVolumeData;
    tsVol.urlTimeSeriesData = urlTimeSeriesData;

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

    ColSch_initColorSchemeGUI(minValue, maxValue, colorRedraw);
    tsVol.currentTimePoint = 0;
    // Update the data shared with the SVG Time Series Fragment
    updateTSFragment();

    // Fire the memory cleaning procedure
    window.setInterval(freeBuffer, tsVol.playbackRate * 20);

    // Start the SVG Time Series Fragment and draw it.
    TSF_initVisualizer(tsVol.urlTimeSeriesData);
    drawGraphs();
}
// ==================================== INITIALIZATION CODE END =============================================

// ==================================== DRAWING FUNCTIONS START =============================================

/**
 * Update function necessary for the color picking function
 */
function colorRedraw(){
    drawSceneFunctional(tsVol.currentTimePoint);
}

/**
 * Draws the current view depending on the selected entity
 * @param tIndex The time point we want to draw
 */
function drawSceneFunctional(tIndex) {
    // if we pass no tIndex the function will play
    // from the tsVol.currentTimePoint incrementing it by 1 or going back to 0.
    if(tIndex == null){
        tIndex = tsVol.currentTimePoint;
        tsVol.currentTimePoint++;
        tsVol.currentTimePoint = tsVol.currentTimePoint % tsVol.timeLength;
    }
    updateTSFragment();
    // An array containing the view for each plane.
    var sliceArray = getViewAtTime(tIndex);
    var selectedEntityValue = getSelectedEntityValue(sliceArray);
    TSV_drawVolumeScene(sliceArray, tsVol.selectedEntity, selectedEntityValue);
    updateMoviePlayerSlider();
}

// ==================================== DRAWING FUNCTIONS  END  =============================================

// ==================================== PRIVATE FUNCTIONS START =============================================

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
                if(privateID === tsVol.batchID){
                    parseAsync(response, function(json){
                        // save the parsed JSON
                        tsVol.bufferL2[sect] = json;
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
    if(tsVol.requestQueue.length < 2) {
        var currentSection = Math.ceil(tsVol.currentTimePoint/tsVol.bufferSize);
        var maxSections = Math.floor(tsVol.timeLength/tsVol.bufferSize);
        var xPlane = ";x_plane=" + (tsVol.selectedEntity[0]);
        var yPlane = ";y_plane=" + (tsVol.selectedEntity[1]);
        var zPlane = ";z_plane=" + (tsVol.selectedEntity[2]);

        for (var i = 0; i <= tsVol.lookAhead && i < maxSections; i++) {
            var toBufferSection = Math.min( currentSection + i, maxSections );
            // If not already requested:
            if(!tsVol.bufferL2[toBufferSection] && tsVol.requestQueue.indexOf(toBufferSection) < 0) {
                var from = toBufferSection * tsVol.bufferSize;
                var to = Math.min(from + tsVol.bufferSize, tsVol.timeLength);
                var query = tsVol.urlVolumeData + "from_idx=" + from + ";to_idx=" + to + xPlane + yPlane + zPlane;
                asyncRequest(query, toBufferSection);
                return; // break out of the loop
            }
        }
    }
}

/**
 *  This function is called to erase some elements from bufferL2 array and avoid
 *  consuming too much memory.
 */
function freeBuffer() {
    var section = Math.floor(tsVol.currentTimePoint/tsVol.bufferSize);
    var bufferedElements = Object.keys(tsVol.bufferL2).length;

    if(bufferedElements > tsVol.bufferL2Size){
        for(var idx in tsVol.bufferL2){
            if (idx < (section - tsVol.bufferL2Size/2) % tsVol.timeLength || idx > (section + tsVol.bufferL2Size/2) % tsVol.timeLength) {
                delete tsVol.bufferL2[idx];
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
 *  This functions returns the X,Y,Z data from time-point t.
 * @param t The time point we want to get
 * @returns Array with only the data from the x,y,z plane at time-point t.
 */
function getViewAtTime(t) {
    var buffer;
    var from;
    var to;
    var xPlane = ";x_plane=" + (tsVol.selectedEntity[0]);
    var yPlane = ";y_plane=" + (tsVol.selectedEntity[1]);
    var zPlane = ";z_plane=" + (tsVol.selectedEntity[2]);

    var query;
    var section = Math.floor(t/tsVol.bufferSize);

    if (tsVol.bufferL2[section]) { // We have that slice in memory
        buffer = tsVol.bufferL2[section];

    } else { // We need to load that slice from the server
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
* Sets tsVol.selectedEntityValue, which represents the selected voxel intensity value, to be highlighted on the legend
*/
function getSelectedEntityValue(sliceArray){
    return sliceArray[0][tsVol.selectedEntity[0]][tsVol.selectedEntity[1]];
}

// ==================================== HELPER FUNCTIONS END ==============================================

// ==================================== PICKING RELATED CODE START ==========================================

function customMouseDown(e){
    e.preventDefault();
    this.mouseDown = true;            // `this` is the canvas
    TSV_pick(e);
}

function customMouseUp(e){
    e.preventDefault();
    this.mouseDown = false;

    if(tsVol.resumePlayer) {
        window.setTimeout(playBack, tsVol.playbackRate * 2);
        tsVol.resumePlayer = false;
    }
    if(tsVol.selectedQuad.index === 3){
        drawGraphs();
    }
}

/**
 * Implements picking and redraws the scene. Updates sliders too.
 * @param e The click event
 */
function TSV_pick(e) {

    if(tsVol.playerIntervalID){
        stopPlayback();
        tsVol.resumePlayer = true;
    }
    var hit = TSV_hitTest(e);
    if (!hit){
        return;
    }
    tsVol.selectedQuad = hit.selectedQuad;
    tsVol.selectedEntity[tsVol.selectedQuad.axes.x] = hit.selectedEntityOnX;
    tsVol.selectedEntity[tsVol.selectedQuad.axes.y] = hit.selectedEntityOnY;
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
    drawSceneFunctional(tsVol.currentTimePoint);
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
    if(!tsVol.playerIntervalID) {
        tsVol.playerIntervalID = window.setInterval(drawSceneFunctional, tsVol.playbackRate);
    }
    $("#btnPlay")[0].setAttribute("class", "action action-pause");
    startBuffering();
}

function stopPlayback(){
    window.clearInterval(tsVol.playerIntervalID);
    tsVol.playerIntervalID = null;
    $("#btnPlay")[0].setAttribute("class", "action action-run");
    stopBuffering();
}

function togglePlayback() {
    if(!tsVol.playerIntervalID) {
        playBack();
    } else {
        stopPlayback();
    }
}

function startBuffering() {
    // Only start buffering id the computed buffer length > 1. Whe only 1 step can be retrieved it is not worthy,
    // and we will have duplicate retrievals generated
    if(!tsVol.streamToBufferID && tsVol.bufferSize > 1) {
        tsVol.batchID++;
        tsVol.requestQueue = [];
        tsVol.bufferL2 = {};
        tsVol.streamToBufferID = window.setInterval(streamToBuffer, 0);
    }
}

function stopBuffering() {
    window.clearInterval(tsVol.streamToBufferID);
    tsVol.streamToBufferID = null;
}

function playNextTimePoint(){
    tsVol.currentTimePoint++;
    tsVol.currentTimePoint = tsVol.currentTimePoint%(tsVol.timeLength);
    drawSceneFunctional(tsVol.currentTimePoint);
}

function playPreviousTimePoint(){
    if(tsVol.currentTimePoint === 0){
        tsVol.currentTimePoint = tsVol.timeLength;
    }
    drawSceneFunctional(--tsVol.currentTimePoint);
}

function seekFirst(){
    tsVol.currentTimePoint = 0;
    drawSceneFunctional(tsVol.currentTimePoint);
}

function seekEnd(){
    tsVol.currentTimePoint = tsVol.timeLength - 1;
    drawSceneFunctional(tsVol.currentTimePoint - 1);
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
function slideMove(event, ui) {

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
function slideMoved(event, ui) {

    if(tsVol.slidersClicked) {
        tsVol.slidersClicked = false;

        if(tsVol.resumePlayer) {
            tsVol.resumePlayer = false;
            window.setTimeout(playBack, tsVol.playbackRate * 2);
        }
    }
   _coreMoveSliderAxis(event, ui);
}

function _coreMoveSliderAxis(event, ui) {
    var quadID = SLIDERIDS.indexOf(event.target.id);
    var selectedQuad = TSV_getQuadrant([quadID]);

    //  Updates the label value on the slider.
    $("#labelCurrentValueAxis" + SLIDERS[quadID]).empty().text( '[' + ui.value + ']' );
    //  Setup things to draw the scene pointing to the right voxel and redraw it.
    if(quadID === 1) {
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
    drawSceneFunctional(tsVol.currentTimePoint);
}

// ==================================== CALLBACK FUNCTIONS END ===============================================
// ==================================== UI RELATED CODE END ==================================================
