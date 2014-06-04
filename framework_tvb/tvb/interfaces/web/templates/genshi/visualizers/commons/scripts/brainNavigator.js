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

////////////////////////////////////~~~~~~~~START BRAIN NAVIGATOR RELATED CODE~~~~~~~~~~~///////////////////////////

var NAV_navigatorX = 0.0, NAV_navigatorY = 0.0, NAV_navigatorZ = 0.0;
// As we draw the 3D navigator as WebGL object, we need buffers for it.
var NAV_navigatorBuffers = [];
/// When brain Navigator is manipulated and this check is True, projections on X, Y, Z are refreshed at each redraw.
var NAV_inTimeRefresh = false;
var _inTimeRefreshCheckbox;
/// Flag to mark that sections are to be redrawn.
var _redrawSectionView = true;


function NAV_initBrainNavigatorBuffers() {
    NAV_navigatorBuffers[0] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, NAV_navigatorBuffers[0]);
    var vertices = [ // z plane
			        -pointPosition, -pointPosition,  0.0,
			        pointPosition, -pointPosition,  0.0,
			        pointPosition,  pointPosition,  0.0,
			        -pointPosition,  pointPosition,  0.0,
			        // y plane
			        -pointPosition, 0.0, -pointPosition,
			        pointPosition, 0.0, -pointPosition,
			        pointPosition, 0.0,  pointPosition,
			        -pointPosition, 0.0,  pointPosition,
			        // x plane
			        0.0, -pointPosition, -pointPosition,
			        0.0,  pointPosition, -pointPosition,
			        0.0,  pointPosition,  pointPosition,
			        0.0, -pointPosition,  pointPosition
			    ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    NAV_navigatorBuffers[0].itemSize = 3;
    NAV_navigatorBuffers[0].numItems = 8;

    NAV_navigatorBuffers[1] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, NAV_navigatorBuffers[1]);
    var vertexNormals = [ // z plane
				        0.0,  0.0,  1.0,
				        0.0,  0.0,  1.0,
				        0.0,  0.0,  1.0,
				        0.0,  0.0,  1.0,
				        // y plane
				        0.0, -1.0,  0.0,
				        0.0, -1.0,  0.0,
				        0.0, -1.0,  0.0,
				        0.0, -1.0,  0.0,
				        // x plane
				        1.0,  0.0,  0.0,
				        1.0,  0.0,  0.0,
				        1.0,  0.0,  0.0,
				        1.0,  0.0,  0.0
				    ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertexNormals), gl.STATIC_DRAW);
    NAV_navigatorBuffers[1].itemSize = 3;
    NAV_navigatorBuffers[1].numItems = 12;

    NAV_navigatorBuffers[2] = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, NAV_navigatorBuffers[2]);
    var cubeVertexIndices = [   0, 1, 2,      0, 2, 3,    // z plane
						        4, 5, 6,      4, 6, 7,    // y plane
						        8, 9, 10,     8, 10, 11  // x plane
						    ];
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(cubeVertexIndices), gl.STATIC_DRAW);
    NAV_navigatorBuffers[2].itemSize = 1;
    NAV_navigatorBuffers[2].numItems = 18;
    // Fake buffers, these won't be used, they only need to be passed
    if (isOneToOneMapping) {
    	var same_color = [];
        for (var i=0; i<NAV_navigatorBuffers[0].numItems* 4; i++) {
        	same_color = same_color.concat(0.34, 0.95, 0.37, 1.0);
        }
        NAV_navigatorBuffers[3] = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, NAV_navigatorBuffers[3]);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(same_color), gl.STATIC_DRAW);
    } else {
	    NAV_navigatorBuffers[3] = NAV_navigatorBuffers[1];
	    NAV_navigatorBuffers[4] = NAV_navigatorBuffers[1];
	}
}

function NAV_setInTimeRefresh(checkbox) {
	_inTimeRefreshCheckbox = checkbox;
	NAV_inTimeRefresh = checkbox.checked;
}

function NAV_customMouseUp(event) {
    GL_handleMouseUp(event);
	NAV_inTimeRefresh = _inTimeRefreshCheckbox && _inTimeRefreshCheckbox.checked;
}

////////////////////////////////////~~~~~~~~END BRAIN NAVIGATOR RELATED CODE~~~~~~~~~~~/////////////////////////////

function NAV_draw_navigator() {
	mvTranslate([NAV_navigatorX, NAV_navigatorY, NAV_navigatorZ]);
	if (_redrawSectionView || NAV_inTimeRefresh) {
        drawSectionView('x', false);
        drawSectionView('y', false);
        drawSectionView('z', false);
        _redrawSectionView = false;
    }
	drawBuffers(gl.TRIANGLES, [NAV_navigatorBuffers], null, true);
}

////////////////////////////////////~~~~~~~~START BRAIN SECTION VIEW RELATED CODE~~~~~~~~~~~///////////////////////////

function drawSectionView(axis, first) {
    var sectionViewRotationMatrix = Matrix.I(4);
    var sectionValue;
    if (axis != undefined) {
        if (axis == 'x') {
            sectionViewRotationMatrix = createRotationMatrix(90, [0, 1, 0]).x(createRotationMatrix(270, [1, 0, 0]));
            sectionValue = NAV_navigatorX;
        }
        if (axis == 'y') {
            sectionViewRotationMatrix = createRotationMatrix(90, [1, 0, 0]).x(createRotationMatrix(180, [1, 0, 0]));
            sectionValue = NAV_navigatorY;
        }
        if (axis == 'z') {
            sectionViewRotationMatrix = createRotationMatrix(180, [0, 1, 0]).x(Matrix.I(4));
            sectionValue = NAV_navigatorZ;
        }
    }
    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    perspective(fov, gl.viewportWidth / gl.viewportHeight, 250 + sectionValue - 1, 250 + sectionValue);
    loadIdentity();
    mvTranslate([0.0, -5.0, -250.0]);
    mvPushMatrix();
    multMatrix(sectionViewRotationMatrix);
    drawBuffers(gl.TRIANGLES, brainBuffers);
    mvPopMatrix();
    displaySection(BRAIN_CANVAS_ID, 'brain-' + axis, axis, first);

    var brainAreaLabel = NAV_getAreaLabel([NAV_navigatorX, NAV_navigatorY, NAV_navigatorZ], measurePoints, measurePointsLabels);
    $(".brainArea").text("Brain area: " + brainAreaLabel);
    //gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

////////////////////////////////////~~~~~~~~END BRAIN SECTION VIEW RELATED CODE~~~~~~~~~~~/////////////////////////////

/**
 * Used for calculating the distance between two points
 */
function _distance(point1, point2) {
    return Math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]) + (point1[2] - point2[2]) * (point1[2] - point2[2]));
}

/**
 * @param point a point of form [x, y, z]
 * @param positions represents a list of points
 */
function _findClosestPosition(point, positions) {
    if (positions == undefined || positions.length == 0) {
        displayMessage("Invalid position parameters passed...", "warningMessage");
    }
    var minDistance = _distance(point, positions[0]);
    var closestPosition = 0;
    for (var pp = 0; pp < positions.length; pp++) {
        var dist = _distance(point, positions[pp]);
        if (dist <= minDistance) {
            minDistance = dist;
            closestPosition = pp;
        }
    }
    return closestPosition;
}

/**
 * @param point a point ([x, y, z]) for which we want to find its area label
 * @param measurePoints an array which contains the measure points
 * @param measurePointsLabels an array which contains the labels for the measure points
 */
function NAV_getAreaLabel(point, measurePoints, measurePointsLabels) {
    var undefinedAreaStr = "Area undefined";
    if (measurePointsLabels == undefined || measurePointsLabels.length == 0) {
        return undefinedAreaStr;
    }
    var closestPosition = _findClosestPosition(point, measurePoints);
    if (measurePointsLabels.length > closestPosition) {
        return measurePointsLabels[closestPosition];
    }
    return undefinedAreaStr;
}



// ------------------------------------- START CODE FOR PICKING FROM 2D NAVIGATOR SECTIONS ----------------------------
/**
 * Find the intersection between a point given by mouseX and mouseY coordinates and a plane
 * given by an axis represented by type (0=x, 1=y, 2=z)
 */
function _findPlaneIntersect(mouseX, mouseY, mvPickMatrix, near, aspect, fov, type) {

  	var centered_x, centered_y, unit_x, unit_y, near_height, near_width, i, j;

	centered_y = gl.viewportHeight - mouseY - gl.viewportHeight/2.0;
	centered_x = mouseX - gl.viewportWidth/2.0;
	unit_x = centered_x/(gl.viewportWidth/2.0);
	unit_y = centered_y/(gl.viewportHeight/2.0);
	near_height = near * Math.tan( fov * Math.PI / 360.0 );
	near_width = near_height*aspect;
	var ray = Vector.create([ unit_x*near_width, unit_y*near_height, -1.0*near, 0 ]);
	var ray_start_point = Vector.create([ 0.0, 0.0, 0.0, 1.0 ]);

	//mvPickMatrix = translateMatrix(currentPoint, mvPickMatrix);
	var R = mvPickMatrix.minor(1,1,3,3);
	var Rt = R.transpose();
	var tc = mvPickMatrix.col(4);
	var t = Vector.create([ tc.e(1), tc.e(2), tc.e(3) ]);
	var tp = Rt.x(t);
	var mvPickMatrixInv = Matrix.I(4);
	for (i=0; i < 3; i++) {
		for (j=0; j < 3; j++) {
			mvPickMatrixInv.elements[i][j] = Rt.elements[i][j];
		}
		mvPickMatrixInv.elements[i][3] = -1.0 * tp.elements[i];
	}
	var rayp = mvPickMatrixInv.x(ray);
	var ray_start_pointp = mvPickMatrixInv.x(ray_start_point);
	var anchor = Vector.create([ ray_start_pointp.e(1), ray_start_pointp.e(2), ray_start_pointp.e(3) ]);
	var direction = Vector.create([ rayp.e(1), rayp.e(2), rayp.e(3) ]);

    // Perform intersection test between ray l and world geometry (cube)
    // Line and Plane objects are taken from sylvester.js library
	var l = Line.create(anchor, direction.toUnitVector());
	// Geometry in this case is a front plane of cube at z = 1;
	anchor = Vector.create([ 0, 0, 0 ]);    // Vertex of the cube
	var normal = Vector.create([0, 0, 0 ]);  // Normal of front face of cube
	normal.elements[type] = 1;
    return _getIntersection(anchor, normal, l);
}

function _getIntersection(anchor, normal, l) {
	var p = Plane.create(anchor, normal);   // Plane
	// Check if line l and plane p intersect
	if (l.intersects(p)) {
		var intersectionPt = l.intersectionWith(p);
		return intersectionPt.elements;
	}
	return null;
}

/**
 * The handlers for the 3 navigator windows. For each of them the steps are:
 * 1. Get mouse position
 * 2. Check if click was done on the IMG element or in the rest of the div(ignore the later)
 * 3. Apply the basic transformations that were done in generating the navigator IMG
 * 4. Find intersection with specific plane and move navigator there
 */
function _handleAxePick(event, axe) {
	// Get the mouse position relative to the canvas element.
    var GL_mouseXRelToCanvasImg = 0;
    var GL_mouseYRelToCanvasImg = 0;
    if (event.offsetX || event.offsetX == 0) { // Opera and Chrome
        GL_mouseXRelToCanvasImg = event.offsetX;
        GL_mouseYRelToCanvasImg = event.offsetY;
    } else if (event.layerX || event.layerY == 0) { // Firefox
       GL_mouseXRelToCanvasImg = event.layerX;
       GL_mouseYRelToCanvasImg = event.layerY;
    }
    if ((event.originalTarget != undefined && event.originalTarget.nodeName == 'IMG') || (
    	event.srcElement != undefined && event.srcElement.nodeName == 'IMG')) {
        var glCanvasElem = $('#GLcanvas');
        GL_mouseXRelToCanvas = (GL_mouseXRelToCanvasImg * glCanvasElem.width()) / 250.0;
        GL_mouseYRelToCanvas = (GL_mouseYRelToCanvasImg * glCanvasElem.height()) / 172.0;

    	perspective(45, gl.viewportWidth / gl.viewportHeight, near, 800.0);
        loadIdentity();
        basicAddLight(lightSettings);

        // Translate to get a good view.
        mvTranslate([0.0, -5.0, -GL_DEFAULT_Z_POS]);
        mvRotate(180, [0, 0, 1]);

        var sectionViewRotationMatrix, result;
        if (axe == 'x') {
            sectionViewRotationMatrix = createRotationMatrix(90, [0, 1, 0]).x(createRotationMatrix(270, [1, 0, 0]));
            multMatrix(sectionViewRotationMatrix);
            result = _findPlaneIntersect(GL_mouseXRelToCanvas, GL_mouseYRelToCanvas, GL_mvMatrix, near, gl.viewportWidth / gl.viewportHeight, fov, 0);
            NAV_navigatorY = result[1];
            NAV_navigatorZ = -result[2];
        } else if (axe == 'y'){
            sectionViewRotationMatrix = createRotationMatrix(90, [1, 0, 0]).x(createRotationMatrix(180, [1, 0, 0]));
            multMatrix(sectionViewRotationMatrix);
            result = _findPlaneIntersect(GL_mouseXRelToCanvas, GL_mouseYRelToCanvas, GL_mvMatrix, near, gl.viewportWidth / gl.viewportHeight, fov, 1);
            NAV_navigatorX = result[0];
            NAV_navigatorZ = -result[2];
        } else {
            sectionViewRotationMatrix = createRotationMatrix(180, [0, 1, 0]).x(Matrix.I(4));
            multMatrix(sectionViewRotationMatrix);
            result = _findPlaneIntersect(GL_mouseXRelToCanvas, GL_mouseYRelToCanvas, GL_mvMatrix, near, gl.viewportWidth / gl.viewportHeight, fov, 2);
            NAV_navigatorX = -result[0];
            NAV_navigatorY = result[1];
        }
		_redrawSectionView = true;
		NAV_draw_navigator();
    }
}


function handleXLocale(event) {
    _handleAxePick(event, 'x');
}

function handleYLocale(event) {
    _handleAxePick(event, 'y');
}

function handleZLocale(event) {
    _handleAxePick(event, 'z');
}
