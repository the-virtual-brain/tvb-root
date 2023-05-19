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

// ------ MOUSE FUNCTIONS -----------------------------------------------------
var TRANSLATION_SENSITIVITY = 4; // screen pixels for a unit
var ROTATION_SENSITIVITY = 20;   // screen pixel per degree
var WHEEL_SENSITIVITY = 0.1;

// -- Globals implementing the Trackball style navigation. --
// The math is camera * trackBall * model * x.  trackball = R2*T2*R1*T1...
// The "view" rotations happen in the trackball matrix. Why?
// We want translations to happen in camera space. This variation: camera = R2R1.. trackball = T2T1..
// will rotate the camera around fine but the translations will happen in model space.
// This is not intuitive. Dragging to the right will move in model +x but model +x can point in arbitrary
// directions (including left) in camera space.

// From model space to trackball space. Modified by mouse left and right button drags.
var GL_trackBallMatrix = Matrix.I(4);
// From trackball space to camera space; Modified by mouse scrolls and middle button drags.
var GL_cameraMatrix = Matrix.Translation($V([0, 0, -200]));

GL_mvMatrix = GL_cameraMatrix.x(GL_trackBallMatrix);

// Which mouse button is pressed 0,1,2,3 none, left, middle, right.
// Here because mouse move event properties (button, buttons, which) are not reliable. Set on mousedown, unset to 0 on mouseup
// Also it enables us to register document wide mouse move event to handle dragging outside the canvas.
var GL_mouseDown = 0;
var GL_lastMouseX = null;
var GL_lastMouseY = null;

var GL_mouseXRelToCanvas = null;
var GL_mouseYRelToCanvas = null;

var ROTATION_EVENT_ID = null;

function GL_handleMouseDown(event, canvas) {
    GL_mouseDown = event.which;
    GL_lastMouseX = event.clientX;
    GL_lastMouseY = event.clientY;

    // Get the mouse position relative to the canvas element.
    var canvasOffset = $(canvas).offset();
    GL_mouseXRelToCanvas = GL_lastMouseX + document.body.scrollLeft + document.documentElement.scrollLeft - Math.floor(canvasOffset.left);
    GL_mouseYRelToCanvas = GL_lastMouseY + document.body.scrollTop + document.documentElement.scrollTop - Math.floor(canvasOffset.top) + 1;

    // Dragging in the canvas should not start a selection in the document
    event.preventDefault();
    // A default we want back is focus on click; to receive keyboard events
    event.target.focus();
}

function GL_handleMouseUp(event) {
    GL_mouseDown = 0;
}

/**
 * Implements a trackball-ish navigation for the scene.
 * Left button rotates in trackball space. Right one translates.
 * Middle mouse moves camera closer/ further from model.
 * Shift + left mouse == right mouse
 * Ctrl will rotate/translate in model space.
 */
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
    // event.buttons not event.button. Latter is 0 in mouse moves in the w3c and mozilla
    var shouldZoomCamera = GL_mouseDown === 2;  // middle click
    var movement;

    if (shouldZoomCamera) { //camera input
        movement = Matrix.Translation($V([0, 0, -deltaY / TRANSLATION_SENSITIVITY]));
        GL_cameraMatrix = movement.x(GL_cameraMatrix);
    } else { // trackball input
        var shouldTranslateXY = GL_mouseDown === 3 || event.shiftKey; // right click or shift
        var inModelSpace = event.ctrlKey;

        if (shouldTranslateXY) {
            movement = Matrix.Translation($V([deltaX / TRANSLATION_SENSITIVITY, -deltaY / TRANSLATION_SENSITIVITY, 0]));
        } else { // rotate
            movement = createRotationMatrix(deltaX / ROTATION_SENSITIVITY, [0, 1, 0]);
            movement = movement.x(createRotationMatrix(deltaY / ROTATION_SENSITIVITY, [1, 0, 0]));
        }

        if (!inModelSpace) { // Normal mode. In camera space
            GL_trackBallMatrix = movement.x(GL_trackBallMatrix);
        } else { // model space
            GL_trackBallMatrix = GL_trackBallMatrix.x(movement);
        }
    }

    GL_mvMatrix = GL_cameraMatrix.x(GL_trackBallMatrix);
}

// ------ MOUSE FUNCTIONS END -----------------------------------------------------

// ------ KEY FUNCTIONS -----------------------------------------------------------

function GL_handleKeyDown(event) {
    var processed = true;
    if (String.fromCharCode(event.keyCode) === " ") {
        GL_trackBallMatrix = Matrix.I(4);
    } else if (event.keyCode === 37) {
        //Left cursor key
        GL_trackBallMatrix = createRotationMatrix(270, [1, 0, 0]).x(createRotationMatrix(270, [0, 0, 1]));
    } else if (event.keyCode === 39) {
        //Right cursor key
        GL_trackBallMatrix = createRotationMatrix(270, [1, 0, 0]).x(createRotationMatrix(90, [0, 0, 1]));
    } else if (event.keyCode === 38) {
        // Up cursor key
        GL_trackBallMatrix = createRotationMatrix(270, [1, 0, 0]);
    } else if (event.keyCode === 40) {
        // Down cursor key
        GL_trackBallMatrix = createRotationMatrix(270, [1, 0, 0]).x(createRotationMatrix(180, [0, 0, 1]));
    } else if (event.keyCode === 49) {
        // Key for number 1
        // TVB-2498 Save here the tilted head position
        GL_trackBallMatrix = createRotationMatrix(270, [1, 0, 0]).x(createRotationMatrix(180, [0, 0, 1])
        ).x(createRotationMatrix(30, [0, 1, 0])).x(createRotationMatrix(-45, [0, 0, 1])
        ).x(createRotationMatrix(-15, [0, 1, 0])).x(createRotationMatrix(-10, [0, 0, 1])
        ).x(createRotationMatrix(20, [1, 0, 0])).x(createRotationMatrix(-80, [0, 0, 1])
        ).x(createRotationMatrix(10, [1, 0, 0]));
    } else if (event.keyCode === 80) {
        // P (from play rotation movie)
        ROTATION_EVENT_ID = setInterval(tickRotation, TICK_STEP * 10);
    } else if (event.keyCode === 83) {
        // S (from stop rotation movie)
        clearInterval(ROTATION_EVENT_ID);
    } else {
        processed = false;
    }

    if (processed) {
        GL_cameraMatrix = Matrix.Translation($V([0, 0, -200]));
        GL_mvMatrix = GL_cameraMatrix.x(GL_trackBallMatrix);
        event.preventDefault();
        return false;
    }
}

function tickRotation() {
    GL_mvMatrix = GL_mvMatrix.x(createRotationMatrix(3, [0, 0, 1]));
    drawScene();
}

function GL_handleKeyUp(event) {
}

function GL_handleMouseWeel(delta) {
    var movement = Matrix.Translation($V([0, 0, delta / WHEEL_SENSITIVITY]));
    GL_cameraMatrix = movement.x(GL_cameraMatrix);
    GL_mvMatrix = GL_cameraMatrix.x(GL_trackBallMatrix);
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

