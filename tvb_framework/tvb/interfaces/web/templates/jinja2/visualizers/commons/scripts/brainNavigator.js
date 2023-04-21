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
 * Depends on the following GLOBALS: gl, createRotationMatrix, drawBuffers, loadIdentity, multMatrix,
 *                      mvPushMatrix, mvPopMatrix, mvTranslate, mvRotate, perspective, displaySection
 *
 * @param isOneToOneMapping
 * @param brainBuffers
 * @param measurePoints
 * @param measurePointsLabels
 * @constructor
 */
function NAV_BrainNavigator(isOneToOneMapping, brainBuffers, measurePoints, measurePointsLabels) {
    this.positionX = 0.0;
    this.positionY = 0.0;
    this.positionZ =  0.0;

    this.drawingBuffers = [];
    this.isInTimeRefresh = false;
    this.cacheInTimeRefreshValue = false;
    this.shouldRedrawSections = 3;

    // Cache from the current component to avoid hidden references towards globals.
    this.brainBuffers_REF = brainBuffers;
    this.measurePoints_REF = measurePoints;
    this.measurePointsLabels_REF = measurePointsLabels;

    this.sectionRotationMatrice = [
        createRotationMatrix(90, [0, 1, 0]).x(createRotationMatrix(270, [1, 0, 0])),
        createRotationMatrix(90, [1, 0, 0]).x(createRotationMatrix(180, [1, 0, 0])),
        createRotationMatrix(180, [0, 1, 0]).x(Matrix.I(4)) ];


    this._init = function (isOneToOneMapping) {
        // I. build the Vertices Buffer:
        this.drawingBuffers[0] = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.drawingBuffers[0]);
        var pointPosition = 75.0;
        var vertices = [-pointPosition, -pointPosition,  0.0,   // z plane
                         pointPosition, -pointPosition,  0.0,
                         pointPosition,  pointPosition,  0.0,
                        -pointPosition,  pointPosition,  0.0,
                        -pointPosition, 0.0, -pointPosition,    // y plane
                         pointPosition, 0.0, -pointPosition,
                         pointPosition, 0.0,  pointPosition,
                        -pointPosition, 0.0,  pointPosition,
                        0.0, -pointPosition, -pointPosition,     // x plane
                        0.0,  pointPosition, -pointPosition,
                        0.0,  pointPosition,  pointPosition,
                        0.0, -pointPosition,  pointPosition
                    ];
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        this.drawingBuffers[0].itemSize = 3;
        this.drawingBuffers[0].numItems = 8;

        // II. Build the buffer with normals:
        this.drawingBuffers[1] = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.drawingBuffers[1]);
        var vertexNormals = [   0.0,  0.0,  1.0,    // z plane
                                0.0,  0.0,  1.0,
                                0.0,  0.0,  1.0,
                                0.0,  0.0,  1.0,
                                0.0, -1.0,  0.0,    // y plane
                                0.0, -1.0,  0.0,
                                0.0, -1.0,  0.0,
                                0.0, -1.0,  0.0,
                                1.0,  0.0,  0.0,    // x plane
                                1.0,  0.0,  0.0,
                                1.0,  0.0,  0.0,
                                1.0,  0.0,  0.0 ];
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertexNormals), gl.STATIC_DRAW);
        this.drawingBuffers[1].itemSize = 3;
        this.drawingBuffers[1].numItems = 12;

        // II. Triangles buffer:
        this.drawingBuffers[2] = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.drawingBuffers[2]);
        var cubeVertexIndices = [   0, 1, 2,      0, 2, 3,      // z plane
                                    4, 5, 6,      4, 6, 7,      // y plane
                                    8, 9, 10,     8, 10, 11 ];  // x plane

        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(cubeVertexIndices), gl.STATIC_DRAW);
        this.drawingBuffers[2].itemSize = 1;
        this.drawingBuffers[2].numItems = 18;

        // IV. Fake buffers, these won't be used, they only need to be passed
        if (isOneToOneMapping) {
            var same_color = [];
            for (var i=0; i<this.drawingBuffers[0].numItems* 4; i++) {
                same_color = same_color.concat(0.34, 0.95, 0.37, 1.0);
            }
            this.drawingBuffers[3] = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, this.drawingBuffers[3]);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(same_color), gl.STATIC_DRAW);
        } else {
            this.drawingBuffers[3] = this.drawingBuffers[1];
        }
    };


    this.getPosition = function () {
        return [this.positionX, this.positionY, this.positionZ];
    };

    this.setInTimeRefresh = function (checkbox) {
        this.isInTimeRefresh = checkbox.checked;
        this.cacheInTimeRefreshValue = this.isInTimeRefresh;
    };

    this.temporaryDisableInTimeRefresh = function() {
        this.cacheInTimeRefreshValue = this.isInTimeRefresh;
        this.isInTimeRefresh = false;
    };

    this.endTemporaryDisableInTimeRefresh = function() {
        this.isInTimeRefresh = this.cacheInTimeRefreshValue;
    };


    this.maybeRefreshSections = function () {
        if (this.shouldRedrawSections > 0 || this.isInTimeRefresh) {
            for(var i = 0; i < 3; i++) {
                this._drawSection(i);
            }
            var closestArea = this._findClosestAreaName(this.getPosition());
            $(".brainArea").text("Brain area: " + closestArea);

            // Keep shouldRedrawSections flag ON, until the first steps are gone (otherwise we are drawing only black)
            this.shouldRedrawSections--;
        }
    };

    this.drawNavigator = function () {
        mvPushMatrix();
        mvTranslate(this.getPosition());
        drawBuffers(gl.TRIANGLES, [this.drawingBuffers], null, true);
        mvPopMatrix();
    };


    this._drawSection = function (axisIdx) {

        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        var sectionValue = this.getPosition()[axisIdx];
        perspective(45, gl.viewportWidth / gl.viewportHeight, 250 + sectionValue - 1, 250 + sectionValue);

        // We do not care about current scene or camera rotations,
        // we just want to draw the brain around the navigator's position
        mvPushMatrix();
        loadIdentity();
        mvTranslate([0.0, -5.0, -250.0]);
        multMatrix(this.sectionRotationMatrice[axisIdx]);
        drawBuffers(gl.TRIANGLES, this.brainBuffers_REF);
        mvPopMatrix();

        var axeNames = ['x', 'y', 'z'];
        displaySection(BRAIN_CANVAS_ID, 'brain-' + axeNames[axisIdx], axeNames[axisIdx]);
    };


    this._findClosestAreaName = function (point) {
        if (!this.measurePointsLabels_REF || this.measurePointsLabels_REF.length == 0) {
            return "Area undefined";
        }
        var idx = NAV_BrainNavigator.findClosestPosition(point, this.measurePoints_REF);
        return this.measurePointsLabels_REF[idx];
    };


    this._getIntersection = function (anchor, normal, l) {
        var p = Plane.create(anchor, normal);
        if (l.intersects(p)) {  // Check if line l and plane p intersect
            return l.intersectionWith(p).elements;
        }
        return null;
    };

    this._findPlaneIntersect = function (mouseX, mouseY, mvPickMatrix, near, aspect, fov, type) {
        /**
         * Find the intersection between a point given by mouseX and mouseY coordinates and a plane
         * given by an axis represented by type (0=x, 1=y, 2=z)
         */
        var unit_x = (mouseX - gl.viewportWidth / 2.0) / (gl.viewportWidth/2.0);
        var unit_y = (gl.viewportHeight - mouseY - gl.viewportHeight/2.0) / (gl.viewportHeight/2.0);
        var near_height = near * Math.tan( fov * Math.PI / 360.0 );
        var near_width = near_height * aspect;

        var R = mvPickMatrix.minor(1, 1, 3, 3);
        var Rt = R.transpose();
        var tc = mvPickMatrix.col(4);
        var t = Vector.create([ tc.e(1), tc.e(2), tc.e(3) ]);
        var tp = Rt.x(t);
        var mvPickMatrixInv = Matrix.I(4);

        for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 3; j++) {
                mvPickMatrixInv.elements[i][j] = Rt.elements[i][j];
            }
            mvPickMatrixInv.elements[i][3] = -1.0 * tp.elements[i];
        }
        var ray = mvPickMatrixInv.x( Vector.create([ unit_x * near_width, unit_y * near_height, -1.0 * near, 0 ]) );
        var ray_start_point = mvPickMatrixInv.x( Vector.create( [ 0.0, 0.0, 0.0, 1.0 ] ) );
        var anchor = Vector.create([ ray_start_point.e(1), ray_start_point.e(2), ray_start_point.e(3)]);
        var direction = Vector.create([ ray.e(1), ray.e(2), ray.e(3) ]);

        // Perform intersection test between ray l and world geometry (cube)
        // Line and Plane objects are taken from sylvester.js library
        var l = Line.create(anchor, direction.toUnitVector());
        // Geometry in this case is a front plane of cube at z = 1;
        anchor = Vector.create([ 0, 0, 0 ]);        // Vertex of the cube
        var normal = Vector.create([ 0, 0, 0 ]);     // Normal of front face of cube
        normal.elements[type] = 1;
        return this._getIntersection(anchor, normal, l);
    };

    this._moveSection = function (event, axisIdx) {
        /**
         * The handlers for the 3 navigator windows. For each of them the steps are:
         * 1. Get mouse position
         * 2. Check if click was done on the IMG element or in the rest of the div(ignore the later)
         * 3. Apply the basic transformations that were done in generating the navigator IMG
         * 4. Find intersection with specific plane and move navigator there
         */

        // Get the mouse position relative to the clicked canvas element.
        var clickInImgX = 0, clickInImgY = 0;
        if (event.offsetX || event.offsetX == 0) {      // Opera and Chrome
            clickInImgX = event.offsetX;
            clickInImgY = event.offsetY;
        } else if (event.layerX || event.layerY == 0) { // Firefox
           clickInImgX = event.layerX;
           clickInImgY = event.layerY;
        }
        // Convert in coordinates matching the mainCanvas, because we want the NAV positions to be relative to that.
        var mainCanvas = $('#GLcanvas');
        clickInImgX = (clickInImgX * mainCanvas.width()) / 250.0;
        clickInImgY = (clickInImgY * mainCanvas.height()) / 172.0;

        if (event.originalTarget && event.originalTarget.nodeName == 'IMG' || event.srcElement && event.srcElement.nodeName == 'IMG') {

            perspective(45, gl.viewportWidth / gl.viewportHeight, near, 800.0);

            mvPushMatrix();
            loadIdentity();
            mvRotate(180, [0, 0, 1]);
            mvTranslate([0.0, -5.0, -250.0]);
            multMatrix(this.sectionRotationMatrice[axisIdx]);

            var result = this._findPlaneIntersect(clickInImgX, clickInImgY, GL_mvMatrix, near, gl.viewportWidth / gl.viewportHeight, 45, axisIdx);
            this.positionX = -result[0];
            this.positionY = result[1];
            this.positionZ = -result[2];

            mvPopMatrix();

            this.shouldRedrawSections = 1;
            this.maybeRefreshSections();
            this.drawNavigator();
        }
    };

    this.moveInXSection = function (event) {
        this._moveSection(event, 0);
    };

    this.moveInYSection = function (event) {
        this._moveSection(event, 1);
    };

    this.moveInZSection = function (event) {
        this._moveSection(event, 2);
    };


    this._init(isOneToOneMapping);
}


NAV_BrainNavigator.findClosestPosition = function (point, positions) {
        /**
         * @param point a point of form [x, y, z]
         * @param positions represents a list of 3D points (current measurePoints)
         */

        if (positions == undefined || positions.length == 0) {
            displayMessage("Invalid position parameters passed...", "warningMessage");
        }
        var euclidean = function (point1, point2) {
                                return Math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) +
                                    (point1[1] - point2[1]) * (point1[1] - point2[1]) +
                                    (point1[2] - point2[2]) * (point1[2] - point2[2]));
                            };
        var closestPosition = 0, minDistance = euclidean(point, positions[0]);
        for (var pp = 1; pp < positions.length; pp++) {
            var dist = euclidean(point, positions[pp]);
            if (dist <= minDistance) {
                minDistance = dist;
                closestPosition = pp;
            }
        }
        return closestPosition;
    };