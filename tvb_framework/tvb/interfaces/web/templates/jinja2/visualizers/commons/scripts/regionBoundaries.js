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
 * Depends on the following GLOBALS: gl, doAjaxCall, displayMessage, HLPR_createWebGlBuffer, SHADING_Context,
 * regionLinesLightSettings, setLighting
 *
 * @constructor
 */
function RB_RegionBoundariesController(boundariesURL) {

    this.boundariesVisible = false;

    this.boundaryVertexBuffers = [];
    this.boundaryNormalsBuffers = [];
    this.boundaryEdgesBuffers = [];

    this._init = function (boundariesURL)  {

        if (boundariesURL) {
            var SELF = this;
            doAjaxCall({
                url: boundariesURL,
                async: true,
                success: function (data) {
                    data = $.parseJSON(data);
                    var boundaryVertices = data[0];
                    var boundaryEdges = data[1];
                    var boundaryNormals = data[2];
                    for (var i = 0; i < boundaryVertices.length; i++) {
                        SELF.boundaryVertexBuffers.push(HLPR_createWebGlBuffer(gl, boundaryVertices[i], false, false));
                        SELF.boundaryNormalsBuffers.push(HLPR_createWebGlBuffer(gl, boundaryNormals[i], false, false));
                        SELF.boundaryEdgesBuffers.push(HLPR_createWebGlBuffer(gl, boundaryEdges[i], true, false));
                    }
                }
            });
        }
    };

    this.drawRegionBoundaries = function (drawingMode, brainBuffers, isPicking) {

        if (!(this.boundariesVisible && this.boundaryVertexBuffers && this.boundaryNormalsBuffers && this.boundaryEdgesBuffers)) {
            return;
        }

        if (drawingMode !== gl.POINTS) {
            // Usually draw the boundaries with the same color. But in points mode draw them with the vertex colors.
            var lightSettings = setLighting(regionLinesLightSettings);
        }
        gl.lineWidth(3.0);
        // replace the vertex, normal and element buffers from the brain buffer set. Keep the alpha buffers
        var bufferSets = [];
        for (var c = 0; c < brainBuffers.length; c++) {
            var chunk = brainBuffers[c].slice();
            chunk[0] = this.boundaryVertexBuffers[c];
            chunk[1] = this.boundaryNormalsBuffers[c];
            chunk[2] = this.boundaryEdgesBuffers[c];
            bufferSets.push(chunk);
        }
        setMatrixUniforms();
        for (var i = 0; i < bufferSets.length; i++) {
            var buffers = bufferSets[i];
            if(isOneToOneMapping) {
                if (isPicking) {
                    SHADING_Context.surface_pick_draw(GL_shaderProgram, buffers[0], buffers[1], buffers[4], buffers[3], buffers[2], gl.LINES);
                } else {
                    SHADING_Context.one_to_one_program_draw(GL_shaderProgram, buffers[0], buffers[1], buffers[3], buffers[2], gl.LINES);
                }
            }else{
                SHADING_Context.region_program_draw(GL_shaderProgram, buffers[0], buffers[1], buffers[3], buffers[2], gl.LINES);
            }
        }

        if (drawingMode !== gl.POINTS) {
            setLighting(lightSettings); // we've drawn solid colors, now restore previous lighting
        }

    };

    this.toggleBoundariesVisibility = function () {
        this.boundariesVisible = !this.boundariesVisible;
    };

    this._init(boundariesURL);
}