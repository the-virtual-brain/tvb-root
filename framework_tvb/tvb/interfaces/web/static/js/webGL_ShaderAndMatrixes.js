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

var gl;
function initGL(canvas) {
    try {
        gl = canvas.getContext("experimental-webgl", {preserveDrawingBuffer: true});
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
    } catch(e) {
    }
    if (!gl) {
        displayMessage("Could not initialise WebGL, sorry :-(", "errorMessage");
    }
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

var shaderProgram;
function basicInitShaders(fsShader, vsShader) {
    var fragmentShader = getShader(gl, fsShader); 
    var vertexShader = getShader(gl, vsShader); 

    shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        displayMessage("Could not initialise shaders", "errorMessage");
    }
    gl.useProgram(shaderProgram);

    shaderProgram.vertexPositionAttribute = gl.getAttribLocation(shaderProgram, "aVertexPosition");
    gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);
    shaderProgram.vertexNormalAttribute = gl.getAttribLocation(shaderProgram, "aVertexNormal");
	gl.enableVertexAttribArray(shaderProgram.vertexNormalAttribute);

    shaderProgram.pMatrixUniform = gl.getUniformLocation(shaderProgram, "uPMatrix");
    shaderProgram.mvMatrixUniform = gl.getUniformLocation(shaderProgram, "uMVMatrix");
    shaderProgram.nMatrixUniform = gl.getUniformLocation(shaderProgram, "uNMatrix");
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

/**
 * Initializes light uniforms for the standard shader. Used by surface viewers.
 * Ambient light, a directional light and a point light with specular highlights
 */
function basicInitSurfaceLighting(){
    shaderProgram.ambientColorUniform = gl.getUniformLocation(shaderProgram, "uAmbientColor");
    shaderProgram.lightingDirectionUniform = gl.getUniformLocation(shaderProgram, "uLightingDirection");
    shaderProgram.directionalColorUniform = gl.getUniformLocation(shaderProgram, "uDirectionalColor");
    shaderProgram.materialShininessUniform = gl.getUniformLocation(shaderProgram, "uMaterialShininess");
    shaderProgram.pointLightingLocationUniform = gl.getUniformLocation(shaderProgram, "uPointLightingLocation");
    shaderProgram.pointLightingSpecularColorUniform = gl.getUniformLocation(shaderProgram, "uPointLightingSpecularColor");
}

var defaultLightSettings = {
    ambientColor : [0.6, 0.6, 0.5],
    directionalColor : [0.7, 0.7, 0.7],
    lightDirection : Vector.create([0.5, 0, 1]).toUnitVector().flatten(),
    specularColor: [0.8, 0.8, 0.8],
    materialShininess : 30.0,
    pointLocation : [0, -10, -400]
};

// Use these settings if you prefer more accurate activity colors instead of better lighting
var noSpecularLightSettings = {
    ambientColor : [0.6, 0.6, 0.5],
    directionalColor : [0.5, 0.5, 0.5],
    lightDirection : Vector.create([0.5, 0, 1]).toUnitVector().flatten(),
    specularColor: [0.0, 0.0, 0.0],
    materialShininess : 30.0,
    pointLocation : [0, -10, -400]
};

/**
 * Sets the light uniforms
 * @param s : a light settings object
 */
function basicAddLight(s){
    gl.uniform3f(shaderProgram.ambientColorUniform,
                s.ambientColor[0], s.ambientColor[1], s.ambientColor[2]);
    gl.uniform3f(shaderProgram.lightingDirectionUniform,
                s.lightDirection[0], s.lightDirection[1], s.lightDirection[2]);
    gl.uniform3f(shaderProgram.directionalColorUniform,
                s.directionalColor[0], s.directionalColor[1], s.directionalColor[2]);
    gl.uniform3f(shaderProgram.pointLightingLocationUniform,
                s.pointLocation[0], s.pointLocation[1], s.pointLocation[2]);
    gl.uniform3f(shaderProgram.pointLightingSpecularColorUniform,
                s.specularColor[0], s.specularColor[1], s.specularColor[2]);
    gl.uniform1f(shaderProgram.materialShininessUniform,
                s.materialShininess);
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
    gl.uniformMatrix4fv(shaderProgram.pMatrixUniform, false, new Float32Array(_GL_pMatrix.flatten()));
    gl.uniformMatrix4fv(shaderProgram.mvMatrixUniform, false, new Float32Array(GL_mvMatrix.flatten()));

    var normalMatrix = GL_mvMatrix.inverse();
    normalMatrix = normalMatrix.transpose();
    gl.uniformMatrix4fv(shaderProgram.nMatrixUniform, false, new Float32Array(normalMatrix.flatten()));
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

