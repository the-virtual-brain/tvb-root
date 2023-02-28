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

var gl;

function _haveRequiredGLCapabilities(){
    if(gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS) < 1){
        displayMessage("TVB requires at least 1 vertex texture",  "errorMessage");
        return false;
    }
    if( gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS) < 512 ){
        displayMessage("TVB requires at least 512 wide vertex uniform arrays",  "errorMessage");
        return false;
    }
    return true;
}

function initGL(canvas) {
    gl = canvas.getContext("experimental-webgl", {preserveDrawingBuffer: true});
    if (!gl || ! _haveRequiredGLCapabilities()){
        displayMessage("Could not initialise WebGL, sorry :-(", "errorMessage");
        // By default we continue running js. Some gl calls will fail. To fail fast uncomment the throw below.
        // throw "WebGL init";
    }

    var canvasWidth = safeMath(canvas.clientWidth, canvas.width);
    var canvasHeight = safeMath(canvas.clientHeight, canvas.height);
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    gl.viewportWidth = canvasWidth;
    gl.viewportHeight = canvasHeight;
    // Used to compute original mouse position in case of canvas resize
    gl.newCanvasWidth = canvasWidth;
    gl.newCanvasHeight = canvasHeight;

    // interface-like methods used for HiRes figure exporting
    var scaleAndRedraw = function(isSmall) {
        if (isSmall)                            // when it's small, compute the scale to make it big
            this.scale = C2I_EXPORT_HEIGHT / canvas.height;
        else                                    // when is not small, invert scale to bring it back to original size
            this.scale = 1 / this.scale;
        gl.newCanvasWidth  = gl.viewportWidth  = canvas.width  *= this.scale;
        gl.newCanvasHeight = gl.viewportHeight = canvas.height *= this.scale;
        if (canvas.redrawFunctionRef)
            canvas.redrawFunctionRef();
    };
    canvas.drawForImageExport = function() {scaleAndRedraw(true)} ;      // small
    canvas.afterImageExport   = function() {scaleAndRedraw(false)} ;     // big
}

function safeMath(number1, number2) {
	if (number1 == undefined) {
		number1 = 0;
	}
	if (number2 == undefined) {
		number2 = 0;
	}
	return Math.max(number1, number2);
}

// ------ SHADER FUNCTIONS --------------------------------------------------

var GL_shaderProgram;

function createAndUseShader(fsShader, vsShader) {
    var fragmentShader = getShader(gl, fsShader);
    var vertexShader = getShader(gl, vsShader);

    GL_shaderProgram = gl.createProgram();
    gl.attachShader(GL_shaderProgram, vertexShader);
    gl.attachShader(GL_shaderProgram, fragmentShader);
    gl.linkProgram(GL_shaderProgram);

    if (!gl.getProgramParameter(GL_shaderProgram, gl.LINK_STATUS)) {
        displayMessage("Could not initialise shaders", "errorMessage");
    }
    gl.useProgram(GL_shaderProgram);
}

function getShader(gl, id) {
    var shaderScript = document.getElementById(id);
    if (!shaderScript) {
        return null;
    }
    var str = "";
    var k = shaderScript.firstChild;
    while (k) {
        if (k.nodeType == 3) {
            str += k.textContent;
        }
        k = k.nextSibling;
    }
    var shader;
    if (shaderScript.type == "x-shader/x-fragment") {
        shader = gl.createShader(gl.FRAGMENT_SHADER);
    } else if (shaderScript.type == "x-shader/x-vertex") {
        shader = gl.createShader(gl.VERTEX_SHADER);
    } else {
        return null;
    }

    gl.shaderSource(shader, str);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        displayMessage(gl.getShaderInfoLog(shader), "warningMessage");
        return null;
    }
    return shader;
}
// ------ SHADER FUNCTIONS END--------------------------------------------------

// ------ COMMON LIGHTING FUNCTIONS --------------------------------------------

// light coordinates are in camera space
// A directional light and a point light shine from top left quadrant.
var defaultLightSettings = {
    ambientColor : [0.5, 0.5, 0.5],
    directionalColor : [0.5, 0.5, 0.5],
    lightDirection : Vector.create([-0.2, 0.2, 1]).toUnitVector().flatten(),
    specularColor: [0.0, 0.0, 0.0],
    materialShininess : 32.0,
    pointLocation : [-20, 20, 0]
};

// Use these settings if you prefer more accurate activity colors instead of better lighting
var minimalLighting = {
    ambientColor : [0.9, 0.9, 0.9],
    directionalColor : [0.1, 0.1, 0.1],
    lightDirection : Vector.create([-0.2, 0.2, 1]).toUnitVector().flatten(),
    specularColor: [0.02, 0.02, 0.02],
    materialShininess : 30.0,
    pointLocation : [0, -10, -400]
};

var specularLightSettings = {
    specularColor: [0.5, 0.5, 0.5]
};

// With this settings all triangles will have the pure ambient color, it removes lighting.
var pickingLightSettings = {
    ambientColor: [1.0, 1.0, 1.0],
    directionalColor : [0.0, 0.0, 0.0],
    specularColor : [0.0, 0.0, 0.0],
    materialColor : [0.0, 0.0, 0.0, 0.0]
};

// Settings below are specific to virtualBrain.js
var linesLightSettings = {
    ambientColor: [0.2, 0.2, 0.2],
    directionalColor : [0.1, 0.1, 0.1],
    specularColor : [0.0, 0.0, 0.0],
    materialColor: [ 0.3, 0.1, 0.3, 1.0]
};

var regionLinesLightSettings = {
    ambientColor: [0.2, 0.2, 0.2],
    directionalColor : [0.1, 0.1, 0.1],
    specularColor : [0.0, 0.0, 0.0],
    materialColor: [0.7, 0.7, 0.1, 1.0]
};

var blendingLightSettings = {
    ambientColor: [0.4, 0.4, 0.4],
    directionalColor : [0.4, 0.4, 0.4],
    specularColor : [0.0, 0.0, 0.0],
    materialColor: [0.95, 0.95, 0.95, 0.29]
};

var _GL_currentLighting = defaultLightSettings;

/**
 * Sets the light uniforms.
 * @param {object} [s] A object like defaultLightSettings. All keys are optional.
 * @returns {object} The previous light settings
 * Missing values are taken from the defaults NOT from the current lighting settings!
 */
function basicSetLighting(s) {
    s = s || {};
    for(var k in defaultLightSettings){
        if (s[k] == null){
            s[k] = defaultLightSettings[k];
        }
    }
    SHADING_Context.light_set_uniforms(GL_shaderProgram, s);

    var prev = _GL_currentLighting;
    _GL_currentLighting = s;
    return prev;
}

function setLighting(settings) {
    settings = settings || {};
    var useVertexColors = settings.materialColor == null;
    gl.uniform1i(GL_shaderProgram.useVertexColors, useVertexColors);
    if (! useVertexColors){
        gl.uniform4fv(GL_shaderProgram.materialColor, settings.materialColor);
    }
    return basicSetLighting(settings);
}

// ------ COMMON LIGHTING FUNCTIONS END ----------------------------------------

// ------ MATRIX FUNCTIONS -----------------------------------------------------

var GL_mvMatrix = Matrix.I(4);
function loadIdentity() {
    GL_mvMatrix = Matrix.I(4);
}

function multMatrix(m) {
    GL_mvMatrix = GL_mvMatrix.x(m);
}

function mvTranslate(v) {
    var m = Matrix.Translation($V([v[0], v[1], v[2]])).ensure4x4();
    multMatrix(m);
}

function createRotationMatrix(angle, v) {
    var arad = angle * Math.PI / 180.0;
    return Matrix.Rotation(arad, $V([v[0], v[1], v[2]])).ensure4x4();
}

function mvRotate(ang, v) {
    var arad = ang * Math.PI / 180.0;
    var m = Matrix.Rotation(arad, $V([v[0], v[1], v[2]])).ensure4x4();
    multMatrix(m);
}

function mvScale(v){
    var m = Matrix.Diagonal([v[0], v[1], v[2], 1]);
    multMatrix(m);
}

var _GL_pMatrix;
function perspective(fovy, aspect, znear, zfar) {
    _GL_pMatrix = makePerspective(fovy, aspect, znear, zfar);
}

function setMatrixUniforms() {
    var normalMatrix = GL_mvMatrix.inverse();
    normalMatrix = normalMatrix.transpose();
    SHADING_Context.transform_set_uniforms( GL_shaderProgram, _GL_pMatrix.flatten(),
        GL_mvMatrix.flatten(), normalMatrix.flatten()
    );
}

var _GL_mvMatrixStack = [];
function mvPushMatrix(m) {
    if (m) {
        _GL_mvMatrixStack.push(m.dup());
        GL_mvMatrix = m.dup();
    } else {
        _GL_mvMatrixStack.push(GL_mvMatrix.dup());
    }
}

function mvPopMatrix() {
    if (_GL_mvMatrixStack.length == 0) {
        throw "Invalid popMatrix!";
    }
    GL_mvMatrix = _GL_mvMatrixStack.pop();
    return GL_mvMatrix;
}

// ------ MATRIX FUNCTIONS END -----------------------------------------------------

// ------ BUFFER FUNCTIONS START -----------------------------------------------------

/**
 * Upload data to the gpu
 * @param webglBuffer The target gpu buffer
 * @param arrayBuffer The source ArrayBuffer in memory
 */
function uploadBuffer(webglBuffer, arrayBuffer){
    gl.bindBuffer(gl.ARRAY_BUFFER, webglBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, arrayBuffer, gl.STATIC_DRAW);
}