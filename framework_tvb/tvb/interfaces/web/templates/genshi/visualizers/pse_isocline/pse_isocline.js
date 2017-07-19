/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
 * Created by Dan Pop on 5/24/2017.
 */

var Pse_isocline = {
    Matrix2d: Matrix2d
};

// TODO add hover and click events
function pse_isocline_init(matrix_data, matrix_shape, x_min, x_max, y_min, y_max, vmin, vmax) {

    matrix2d_init(matrix_data, matrix_shape, x_min, x_max, y_min, y_max, vmin, vmax, true);
}


function redrawCanvas(base_url, selected_metric) {
    doAjaxCall({
        url: base_url + '/' + selected_metric,
        type: 'POST',
        async: false,
        success: function (data) {
            var dictionar = $.parseJSON(data);
            Matrix2d.data = $.parseJSON(dictionar.matrix_data);
            Matrix2d.vmin = dictionar.vmin;
            Matrix2d.vmax = dictionar.vmax;
            var dimensions = $.parseJSON(dictionar.matrix_shape);
            Matrix2d.n = dimensions[0];
            Matrix2d.m = dimensions[1];
            var interpolatedMatrix = interpolateMatrix(Matrix2d.canvas.clientWidth);
            Matrix2d.data = matrixToArray(interpolatedMatrix);
            ColSch_initColorSchemeComponent(Matrix2d.vmin, Matrix2d.vmax);
            drawCanvas();
        }
    });
}

function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}

function interpolateMatrix(cWidth) {
    var dataMatrix = [];
    for (var i = 0; i < Matrix2d.n; i++) {
        dataMatrix[i] = [];
        for (var j = 0; j < Matrix2d.m; j++) {
            dataMatrix[i][j] = undefined;
        }
    }
    var oldMatrix = Matrix2d.data;
    for (var i = 0; i < Matrix2d.n; i++)
        for (var j = 0; j < Matrix2d.m; j++)
            dataMatrix[i][j] = oldMatrix[i * Matrix2d.m + j];
    //TODO cWidth not initialized
    var ratio = Math.floor((cWidth / 10) / Matrix2d.m);
    ratio = 20;

    var spaceC = Math.floor((Matrix2d.m * ratio) / (Matrix2d.m - 1));
    var spaceL = Math.floor((Matrix2d.n * ratio) / (Matrix2d.n - 1));
    var n = (spaceL - 1) * (Matrix2d.n - 1) + Matrix2d.n;
    var m = (spaceC - 1) * (Matrix2d.m - 1) + Matrix2d.m;

    var array = interpolateArray(dataMatrix[0], n);

    var newMatrix = [];
    for (i = 0; i < n; i++) {
        newMatrix[i] = [];
        for (j = 0; j < m; j++) {
            newMatrix[i][j] = undefined;
        }
    }

    for (i = 0; i < Matrix2d.n; i++) {
        var line = i * spaceL;
        var col = 0;
        newMatrix[line][col] = dataMatrix[i][0];
        col++;
        for (var k = 1; k < Matrix2d.m; k++) {
            for (j = 0; j < spaceC - 1; j++) {
                newMatrix[line][col] = undefined;
                col++;
            }
            newMatrix[line][col] = dataMatrix[i][k];
            col++;
        }
    }
    console.table(newMatrix);
    for (j = 0; j < Matrix2d.m; j++) {
        var column = [];
        for (i = 0; i < Matrix2d.n; i++) {
            column.push(dataMatrix[i][j])
        }
        interpolatedColumn = interpolateArray(column, n);
        for (i = 0; i < n; i++) {
            newMatrix[i][j * spaceC] = interpolatedColumn[i];
        }
    }
    for (i = 0; i < n; i++) {
        var intermediateLine = newMatrix[i].filter(function (element) {
            return element !== undefined;
        });
        newMatrix[i]=interpolateArray(intermediateLine,m);
    }
    Matrix2d.n = n;
    Matrix2d.m = m;
    return newMatrix;
}

function linearInterpolate(before, after, atPoint) {
    return before + (after - before) * atPoint;
}

function interpolateArray(data, fitCount) {
    var newData = [];
    var springFactor = (data.length - 1) / (fitCount - 1);
    newData[0] = data[0]; // for new allocation
    for (var i = 1; i < fitCount - 1; i++) {
        var tmp = i * springFactor;
        var before = Math.floor(tmp).toFixed();
        var after = Math.ceil(tmp).toFixed();
        var atPoint = tmp - before;
        newData[i] = this.linearInterpolate(data[before], data[after], atPoint);
    }
    newData[fitCount - 1] = data[data.length - 1]; // for new allocation
    return newData;
}

function matrixToArray(matrixData) {
    var n = matrixData.length;
    var m = matrixData[0].length;
    var array = [];
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            array[i * m + j] = matrixData[i][j];
        }
    }
    return array;
}