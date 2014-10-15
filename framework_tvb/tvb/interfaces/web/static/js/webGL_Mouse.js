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

// ------ MOUSE FUNCTIONS -----------------------------------------------------

var GL_mouseDown = false;
var GL_lastMouseX = null;
var GL_lastMouseY = null;
var GL_currentRotationMatrix = Matrix.I(4);

var GL_mouseXRelToCanvas = null;
var GL_mouseYRelToCanvas = null;

function GL_handleMouseDown(event, canvas) {
    GL_mouseDown = true;
    GL_lastMouseX = event.clientX;
    GL_lastMouseY = event.clientY;

   // // Get the mouse position relative to the canvas element.
    var canvasOffset = $(canvas).offset();
    GL_mouseXRelToCanvas = GL_lastMouseX + document.body.scrollLeft + document.documentElement.scrollLeft - Math.floor(canvasOffset.left);
    GL_mouseYRelToCanvas = GL_lastMouseY + document.body.scrollTop + document.documentElement.scrollTop - Math.floor(canvasOffset.top) + 1;

    // Dragging in the canvas should not start a selection in the document
    event.preventDefault();
    // A default we want back is focus on click; to receive keyboard events
    event.target.focus();
}

function GL_handleMouseUp(event) {
    GL_mouseDown = false;
}

function GL_handleMouseMove(event) {
    if (!GL_mouseDown) {
        return;
    }
    var newX = event.clientX;
    var newY = event.clientY;
    var deltaX = newX - GL_lastMouseX;
    var deltaY = newY - GL_lastMouseY;
    GL_lastMouseX = newX;
    GL_lastMouseY = newY;

    var newRotationMatrix = createRotationMatrix(deltaX / 10, [0, 1, 0]);
    newRotationMatrix = newRotationMatrix.x(createRotationMatrix(deltaY / 10, [1, 0, 0]));
    GL_currentRotationMatrix = newRotationMatrix.x(GL_currentRotationMatrix);
    loadIdentity();
    mvTranslate([0.0, -5.0, -GL_zTranslation]);
    multMatrix(GL_currentRotationMatrix);
}

// ------ MOUSE FUNCTIONS END -----------------------------------------------------

// ------ KEY FUNCTIONS -----------------------------------------------------------
var GL_DEFAULT_Z_POS = 0;
var GL_zTranslation = 0;
var _GL_currentlyPressedKeys = Object();

function GL_handleKeyDown(event) {
    _GL_currentlyPressedKeys[event.keyCode] = true;
    if (String.fromCharCode(event.keyCode) == " ") {
        GL_currentRotationMatrix = Matrix.I(4);
        GL_zoomSpeed = 0;
        GL_zTranslation = GL_DEFAULT_Z_POS;
    }
    if (event.keyCode == 37) {
        //Left cursor key
        GL_currentRotationMatrix = createRotationMatrix(270, [1, 0, 0]).x(createRotationMatrix(270, [0, 0, 1]));
    }
    if (event.keyCode == 39) {
        //Right cursor key
        GL_currentRotationMatrix = createRotationMatrix(270, [1, 0, 0]).x(createRotationMatrix(90, [0, 0, 1]));
    }
    if (event.keyCode == 38) {
        // Up cursor key
        GL_currentRotationMatrix = createRotationMatrix(180, [1, 0, 0]).x(Matrix.I(4));
    }
    if (event.keyCode == 40) {
        GL_currentRotationMatrix = Matrix.I(4);
    }
    loadIdentity();
    mvTranslate([0.0, -5.0, -GL_zTranslation]);
    multMatrix(GL_currentRotationMatrix);
    event.preventDefault();
	return false;
}

function GL_handleKeyUp(event) {
    _GL_currentlyPressedKeys[event.keyCode] = false;
    event.preventDefault();
	return false;
}

function GL_handleMouseWeel(delta) {
    var GL_zoomSpeed = 0;
	if (delta < 0) {
        GL_zoomSpeed = 0.2;
    } else if (delta > 0) {
        GL_zoomSpeed = -0.2;
    }
    GL_zTranslation -= GL_zoomSpeed * GL_zTranslation;
    loadIdentity();
    mvTranslate([0.0, -5.0, -GL_zTranslation]);
    multMatrix(GL_currentRotationMatrix);
}
// ------ KEY FUNCTIONS END -----------------------------------------------------

// ------ RESHAPE FUNCTIONS START -----------------------------------------------

/**
 * Get the actual size of the canvas after all styles are applied and resizing is done, and
 * update the gl context with these new values.
 */
function updateGLCanvasSize(canvasId) {
    var canvas = $("#" + canvasId);
    var width = canvas.parent().width();
    var height = canvas.parent().height();

    gl.newCanvasWidth = width;
    gl.newCanvasHeight = height;
    gl.clientWidth = width;
    gl.clientHeight = height;
    gl.viewportWidth = width;
    gl.viewportHeight = height;
    canvas.attr("width", width);
    canvas.attr("height", height);
}
// ------ RESHAPE FUNCTIONS END -------------------------------------------------

