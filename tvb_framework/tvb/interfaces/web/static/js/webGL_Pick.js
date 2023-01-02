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

/**
 * Functions for WebGL Viualizers that need node-pick.
 * The number of nodes for mouse-pick needs to be limited.
 */

/* globals gl */

//Constants that will be used in the picking scheme
GL_BACKGROUND = -2;
GL_NOTFOUND = -1;
//A dummy coordinates array that should fill the whole screen with a color. Used to 
//get the returned color values from a click on a specific color
var _DUMMY_COLOR_PICKING_ARRAY = [
    0.0,  1000.0,  0.0,
    -1000.0, -1000.0,  1000.0,
    1000.0, -1000.0,  1000.0,
    0.0,  1000.0,  0.0,
    1000.0, -1000.0,  1000.0,
    1000.0, -1000.0, -1000.0,
    0.0,  100.0,  0.0,
    1000.0, -1000.0, -1000.0,
   -1000.0, -1000.0, -1000.0,
    0.0,  1000.0,  0.0,
   -1000.0, -1000.0, -1000.0,
   -1000.0, -1000.0,  1000.0
];
//color array that holds for each node a unique color                 				
var GL_colorPickerInitColors = [];
//color dictionary used for quick lookups that gives the node index for a given color
var GL_colorPickerMappingDict = {};
//a buffer used to draw the nodes 'behind the scene' and do the picking computations
var GL_colorPickerBuffer;

function GL_initColorPickingData(numberOfObjects) {
    /*
     * Create the color array and color buffer for a given number of 'selectable' objects
     */
    var buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(_DUMMY_COLOR_PICKING_ARRAY), gl.STATIC_DRAW);
    //Compute a increment that will allow to optimally 'spread' the range of available colors
    var TOTAL_COLOR_NR = 255 * 255 * 255;
    var inc = (TOTAL_COLOR_NR - 1)/ numberOfObjects;

    var idx = 0;
    var r, g, b, input_col;

    for (var i = 1; i < TOTAL_COLOR_NR; i += inc) {
        r = parseInt(parseInt(i/255)/255);
        g = parseInt(i/255)%255;
        b = parseInt(i%255);
        input_col = '' + r + g + b;
        GL_colorPickerInitColors.push([r/255, g/255, b/255, 1]);
        GL_colorPickerMappingDict[input_col] = idx;
        idx = idx + 1;
    }
}

/**
 * Do the initializations for a new buffer to be used for color picking/navigator
 */
function GL_initColorPickFrameBuffer() {
    var rttFramebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, rttFramebuffer);
    rttFramebuffer.width = gl.viewportWidth;
    rttFramebuffer.height = gl.viewportHeight;

    var rttTexture = gl.createTexture();
    // Bind to the second texture unit because the first one is used by the color map texture.
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, rttTexture);
    // This full screen texture should not need any filtering. Sensible for NPOT off-screen texture.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    // Clamp to edge. This is a NPOT texture, other modes are not supported.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, rttFramebuffer.width, rttFramebuffer.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    var renderbuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, renderbuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, rttFramebuffer.width, rttFramebuffer.height);

    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, rttTexture, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, renderbuffer);

    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.activeTexture(gl.TEXTURE0);
    GL_colorPickerBuffer = rttFramebuffer;
    GL_colorPickerInitColors = [];
}


/**
 * Get node index currently picked, if any, otherwise undefined.
 * Will convert from mouse coordinated into webGL coordinates, as on canvas resize we do not update webGL projection coordinates.
 */
function GL_getPickedIndex() {

    var pickX = GL_mouseXRelToCanvas;
    if (gl.viewportWidth != gl.newCanvasWidth) {
        // Avoid rounding problems
        pickX = pickX * gl.viewportWidth /  gl.newCanvasWidth;
    }
    var pickY = GL_mouseYRelToCanvas;
    if (gl.viewportHeight !=  gl.newCanvasHeight) {
        pickY = pickY * gl.viewportHeight /  gl.newCanvasHeight;
    }
    var pixelValues = new Uint8Array(4);
    gl.readPixels(pickX, gl.viewportHeight - pickY, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixelValues);
    var dict_key = '' + pixelValues[0] + pixelValues[1] + pixelValues[2];
    if (dict_key == '000') { return GL_BACKGROUND; }
    if (GL_colorPickerMappingDict[dict_key] != undefined) {
        return GL_colorPickerMappingDict[dict_key];
    } else {
        return GL_NOTFOUND;
    }

}


