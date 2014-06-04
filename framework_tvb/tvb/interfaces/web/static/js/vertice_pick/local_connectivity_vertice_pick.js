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
 * necessary from base_vertice_pick.js, or just prefix your functions. (e.g. LCONN_PICK_${function_name}).
 * ---------------------------------------===========================================--------------------------------------
 */


/**
 * 'Overwrite' the drawScene function here to add extra functionality for drawing the legend
 */
function drawScene() {
    if (GL_zoomSpeed != 0) {		// Handle the zoom event before drawing the brain.
        GL_zTranslation -= GL_zoomSpeed * GL_zTranslation;
        GL_zoomSpeed = 0;
    }
    // Use function offered by base_vertice_pick.js to draw the brain.
	BASE_PICK_drawBrain(BASE_PICK_brainDisplayBuffers, noOfUnloadedBrainDisplayBuffers);

   if (noOfUnloadedBrainDisplayBuffers == 0) {      // wait for the data to be loaded, then draw the legend
        loadIdentity();
        if (LEG_legendBuffers.length) drawBuffers(gl.TRIANGLES, [LEG_legendBuffers]);
   }
}


/**
 * Displays a gradient on the surface.
 *
 * @param data_from_server a json object which contains the data needed
 * for drawing a gradient on the surface.
 */
function LCONN_PICK_updateBrainDrawing(data_from_server) {
    data_from_server = $.parseJSON(data_from_server);

    var data = $.parseJSON(data_from_server['data']);
    var minValue = data_from_server['min_value'];
    var maxValue = data_from_server['max_value'];

    BASE_PICK_initLegendInfo(maxValue, minValue);     // setup the legend
    ColSch_initColorSchemeParams(minValue, maxValue, function() {
        LEG_updateLegendColors();
        _updateBrainColors(data, minValue, maxValue);
        drawScene()
    });

    if (BASE_PICK_brainDisplayBuffers.length != data.length) {
        displayMessage("Could not draw the gradient view. Invalid data received from the server.", "errorMessage");
        return;
    }

    _updateBrainColors(data_from_server);
    drawScene();
    displayMessage("Displaying Local Connectivity profile for selected focal point ..." )
}

/**
 * Updates the buffers for drawing the brain, from the specified data
 * @private
 */
function _updateBrainColors(data, minValue, maxValue) {
    for (var i = 0; i < data.length; i++) {
        var colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        var thisBufferColors = new Float32Array(data[i].length * 4);
        getGradientColorArray(data[i], minValue, maxValue, thisBufferColors);
        gl.bufferData(gl.ARRAY_BUFFER, thisBufferColors, gl.STATIC_DRAW);
        BASE_PICK_brainDisplayBuffers[i][3] = colorBuffer;
    }
}

/**
 * In case something changed in the parameters or the loaded local_connectivity is
 * set to None, just use this method to draw the 'default' surface with the gray coloring.
 */
function LCONN_PICK_drawDefaultColorBuffers() {
    if (noOfUnloadedBrainDisplayBuffers != 0) {
        displayMessage("The load operation for the surface data is not completed yet!", "infoMessage");
        return;
    }
	BASE_PICK_buffer_default_color();
    drawScene();
}

