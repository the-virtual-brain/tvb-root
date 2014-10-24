/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and 
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
 *
 * This program is free software; you can redistribute it and/or modify it under 
 * the terms of the GNU General Public License version 2 as published by the Free
 * Software Foundation. This program is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
 * License for more details. You should have received a copy of the GNU General 
 * Public License along with this program; if not, you can download it here
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0
 *
 **/

/*
 * ---------------------------------------===========================================--------------------------------------
 * WARNING: This script is just adding some functionality specific to the stimulus visualization on top of what is defined 
 * in /static/js/vertice_pick/base_vertice_pick.js. As such in all the cases when this script is used, you must first 
 * include base_vertice_pick.js. In case you need to ADD FUNCTIONS here, either make sure you don't "overwrite" something
 * necessary from base_vertice_pick.js, or just prefix your functions. (e.g. STIM_PICK_${function_name}).
 * ---------------------------------------===========================================--------------------------------------
 */
var BUFFER_TIME_STEPS = 8;
var DATA_CHUNK_SIZE = null;
var TICK_STEP = 200;
var maxValue = 0;
var minValue = 0;
var startColor = [0.5, 0.5, 0.5];
var endColor = [1, 0, 0];
var drawTickInterval = null;
var sliderSel = false;
var minTime = 0;
var maxTime = 0;
var displayedStep = 0;
var totalTimeStep = 0;
var currentStimulusData = null;
var nextStimulusData = null;
var asyncLoadStarted = false;
var endReached = false;


/**
 * Start the movie mode visualization of the received data.
 */
function STIM_PICK_setVisualizedData(data) {

	BASE_PICK_isMovieMode = true;
	currentStimulusData = data['data'];
	minTime = data['time_min'];
	maxTime = data['time_max'];
	DATA_CHUNK_SIZE = data['chunk_size'];
    var colorValuesChanged = (maxValue != data['max']);
	maxValue = data['max'];
    minValue = data['min'];

	// If for some reason the draw timeout was not cleared, clear it now.
	if (drawTickInterval != null) {
		clearInterval(drawTickInterval);
	}
	endReached = false;
	// Reset the flags and set a new draw timeout.
	drawTickInterval = setInterval(tick, TICK_STEP);
	displayedStep = 0;
	totalTimeStep = minTime;

	BASE_PICK_initLegendInfo(maxValue, minValue);
    if (colorValuesChanged) {
        ColSch_initColorSchemeParams(minValue, maxValue, LEG_updateLegendColors);
    }

	//Create the slider to display the total time.
	var sliderDiv = document.createElement('DIV');
	sliderDiv.className = "shadow";
	sliderDiv.id = "slider";
	document.getElementById("slider-div").appendChild(sliderDiv);
	
	$("#slider").slider({ min:minTime, max: maxTime, disabled: true,
	            slide: function() {
	            	sliderSel = true;
	                totalTimeStep = $("#slider").slider("option", "value");
	                displayedStep = parseInt((totalTimeStep - minTime) % DATA_CHUNK_SIZE);
	            },
	            stop: function() {
	            	sliderSel = false;
	            } });
}

/**
 * Stop the movie-mode visualization. Reset flags and steps, also remove the
 * tick - timeout and reset the color buffers.
 */
function STIM_PICK_stopDataVisualization() {

    if (noOfUnloadedBrainDisplayBuffers != 0) {
        displayMessage("The load operation for the surface data is not completed yet!", "infoMessage");
        return;
    }
	BASE_PICK_isMovieMode = false;
	document.getElementById('brainLegendDiv').innerHTML = '';
	displayedStep = 0;
	if (drawTickInterval != null) {
		clearInterval(drawTickInterval);
	}
	document.getElementById("slider-div").innerHTML = '';
	document.getElementById("TimeNow").innerHTML = '';
	asyncLoadStarted = false;
    BASE_PICK_buffer_default_color();
    drawScene();
}

/**
 * Since the min-max interval can be quite large, just load it in chunks.
 */
function STIM_PICK_loadNextStimulusChunk() {

	var currentChunkIdx = parseInt((totalTimeStep - minTime) / DATA_CHUNK_SIZE);
	if ((currentChunkIdx + 1) * DATA_CHUNK_SIZE < (maxTime - minTime)) {
		// We haven't reached the final chunk so just load it normally.
		asyncLoadStarted = true;
		doAjaxCall({
			        type:'GET',
			        url:'/spatial/stimulus/surface/get_stimulus_chunk/' + (currentChunkIdx + 1),
			        success:function (data) {
			            nextStimulusData = $.parseJSON(data);
			            asyncLoadStarted = false;
			        },
                    error: function() {
                        displayMessage("Something went wrong in getting the next stimulus chunk!", "warningMessage")
                    }
			    });
	} else {
		// No more chunks to load. Set end of data flat and block the async load by setting
		// asyncLoadStarted to true so no more calls to loadNextStimulusChunk are done for this data.
		asyncLoadStarted = true;
		endReached = true;
	}
}

/**
 * Function called every TICK_STEP milliseconds. This is only done in movie mode.
 */
function tick() {

    if (noOfUnloadedBrainDisplayBuffers != 0) {
        displayMessage("The load operation for the surface data is not completed yet!", "infoMessage");
        return;
    }
	if (! sliderSel ) {
		//If we reached maxTime, the movie ended
        var buttonRun = $('.action-run')[0];
        var buttonStop = $('.action-stop')[0];

		if (BASE_PICK_isMovieMode && totalTimeStep < maxTime) {
			// We are still in movie mode and didn't pass the end of the data.
		    if (displayedStep >= currentStimulusData.length) {
		    	if (currentStimulusData.length > maxTime - minTime || endReached ) {
		    		//We had reached the end of the movie mode.
		    		STIM_PICK_stopDataVisualization();
					buttonRun.className = buttonRun.className.replace('action-idle', '');
					buttonStop.className = buttonStop.className + " action-idle";
		    	} else {
		    		//If the async load of the next data chunk is done, do the switch otherwise just wait
		    		if (nextStimulusData != null) {
		    			currentStimulusData = nextStimulusData;
		    			displayedStep = 0;
		    			nextStimulusData = null;
		    		} else {
		    			return;
		    		}
		    	}
		    }
			var thisStepData = currentStimulusData[displayedStep];
			// Compute the colors for this current step:

			for (var i=0; i<BASE_PICK_brainDisplayBuffers.length;i++) {
				BASE_PICK_brainDisplayBuffers[i][3] = null;
				var upperBorder = BASE_PICK_brainDisplayBuffers[i][0].numItems / 3;
			    var thisBufferColors = new Float32Array(upperBorder * 4);
			    var offset_start = i * 40000;
                getGradientColorArray(thisStepData.slice(offset_start, offset_start + upperBorder),
                                      minValue, maxValue, thisBufferColors);
		    	BASE_PICK_brainDisplayBuffers[i][3] = gl.createBuffer();
		    	gl.bindBuffer(gl.ARRAY_BUFFER, BASE_PICK_brainDisplayBuffers[i][3]);
	            gl.bufferData(gl.ARRAY_BUFFER, thisBufferColors, gl.STATIC_DRAW);
	            thisBufferColors = null;
			}
			//Redraw the scene 
		    drawScene();
		} else {
			//We had reached the end of the movie mode.
    		STIM_PICK_stopDataVisualization();
			buttonRun.className = buttonRun.className.replace('action-idle', '');
			buttonStop.className = buttonStop.className + " action-idle";
		}
	}
}


function drawScene() {
    var theme = ColSchGetTheme().surfaceViewer;
    gl.clearColor(theme.backgroundColor[0], theme.backgroundColor[1], theme.backgroundColor[2], theme.backgroundColor[3]);
    // Use function offered by base_vertice_pick.js to draw the brain:
	BASE_PICK_drawBrain();
	if (BASE_PICK_isMovieMode) {
		// We are in movie mode so drawScene was called automatically. We don't
		// want to update the slices here to improve performance. Increse the timestep.
		displayedStep += 1;
		totalTimeStep += 1;
		if (currentStimulusData.length < (maxTime - minTime) &&
            displayedStep + BUFFER_TIME_STEPS >= currentStimulusData.length &&
            nextStimulusData == null && ! asyncLoadStarted ) {
			STIM_PICK_loadNextStimulusChunk();
		}
		if (!sliderSel) {
            document.getElementById("TimeNow").innerHTML = "Time: " + totalTimeStep + " ms";
            $("#slider").slider("option", "value", totalTimeStep);
        }
		// Draw the legend for the stimulus now.
        mvPushMatrix();
		loadIdentity();
	    basicAddLight(defaultLightSettings);
		drawBuffers(gl.TRIANGLES, [LEG_legendBuffers]);
        mvPopMatrix();
	} else {
		// We are not in movie mode. The drawScene was called from some ui event (e.g.
		// mouse over). Here we can afford to update the 2D slices because performance is
		// not that much of an issue.
		BASE_PICK_drawBrain();
	}
}