/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need to download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

/* globals  GVAR_connectivityMatrix, GVAR_interestAreaVariables, GVAR_selectedAreaType,
            GVAR_pointsLabels, GVAR_interestAreaNodeIndexes, HLPR_removeByElement,
            GFUNC_updateLeftSideVisualization, GFUNC_isNodeAddedToInterestArea,
            GFUN_updateSelectionComponent, GFUNC_toggleNodeInInterestArea,
            displayMessage, doAjaxCall,
            CONN_getLineWidthValue, CONN_lineWidthsBins, CONN_comingInLinesIndices, CONN_comingOutLinesIndices, CONN_initLinesHistorgram,
            NO_POSITIONS, highlightedPointIndex1, highlightedPointIndex2,
            ColSch_getGradientColorString, ColSch_updateLegendColors, ColSch_updateLegendLabels
*/

/*
 * This file handles the display and functionality of the 2d matrix view.
 * The matrix is rendered on a <canvas> (one per quadrant) instead of an N*N HTML table,
 * so it stays fast for large connectivities. All editing/selection logic keeps operating
 * on GVAR_interestAreaVariables, exactly as before.
 */

// Width in css pixels reserved for the row/column label gutters on each canvas.
var MATRIX_LABEL_GUTTER = 66;
// Below these cell sizes labels would be illegible, so we skip them and rely on the hover tooltip.
var MATRIX_MIN_LABEL_CELL = 7;
// Largest canvas height (css pixels); keeps a whole quadrant visible without a giant scroll area.
var MATRIX_MAX_CANVAS_HEIGHT = 640;

/*
 * The cell currently opened for editing, as {prefix, i, j}, or null. Used to draw its highlight border.
 */
var MATRIX_editedCell = null;


/**
 * @return the weights (prefix 'w') or tracts (prefix 't') matrix descriptor.
 */
function _matrixByPrefix(prefix) {
    for (var key in GVAR_interestAreaVariables) {
        if (GVAR_interestAreaVariables[key].prefix === prefix) {
            return GVAR_interestAreaVariables[key];
        }
    }
    return null;
}

/**
 * Truncate text with an ellipsis so it fits within maxWidth for the given canvas context.
 */
function _fitLabel(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) {
        return text;
    }
    while (text.length > 1 && ctx.measureText(text + '…').width > maxWidth) {
        text = text.slice(0, -1);
    }
    return text + '…';
}

/**
 * Read the {colStart, colEnd, rowStart, rowEnd, prefix} a canvas quadrant covers, plus its layout geometry.
 */
function _canvasGeometry(canvas) {
    var d = canvas.dataset;
    var g = {
        prefix: d.prefix,
        colStart: parseInt(d.colStart, 10), colEnd: parseInt(d.colEnd, 10),
        rowStart: parseInt(d.rowStart, 10), rowEnd: parseInt(d.rowEnd, 10)
    };
    g.nCols = g.colEnd - g.colStart;
    g.nRows = g.rowEnd - g.rowStart;
    g.cssW = canvas.clientWidth;
    g.cssH = Math.min(g.cssW, MATRIX_MAX_CANVAS_HEIGHT);
    g.gridW = g.cssW - MATRIX_LABEL_GUTTER;
    g.gridH = g.cssH - MATRIX_LABEL_GUTTER;
    g.cw = g.gridW / g.nCols;
    g.ch = g.gridH / g.nRows;
    return g;
}

/**
 * Draw a single quadrant canvas: cells, selection overlay, labels and the edited-cell border.
 */
function _drawQuadrant(canvas) {
    var g = _canvasGeometry(canvas);
    if (!g.cssW || g.nCols <= 0 || g.nRows <= 0) {
        return;                                             // hidden or empty quadrant
    }

    var matrix = _matrixByPrefix(g.prefix);
    var values = matrix.values;
    if (!values || !values.length) {
        return;
    }
    var minValue = matrix.min_val;
    var maxValue = matrix.max_val;

    var dpr = window.devicePixelRatio || 1;
    canvas.style.height = g.cssH + 'px';
    canvas.width = Math.round(g.cssW * dpr);
    canvas.height = Math.round(g.cssH * dpr);
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, g.cssW, g.cssH);

    // Which rows/cols are part of the interest area (computed once per axis, not per cell).
    var rowSel = [];
    var colSel = [];
    var r, c, i, j, x, y;
    for (r = 0; r < g.nRows; r++) {
        rowSel[r] = GFUNC_isNodeAddedToInterestArea(g.rowStart + r);
    }
    for (c = 0; c < g.nCols; c++) {
        colSel[c] = GFUNC_isNodeAddedToInterestArea(g.colStart + c);
    }

    // Cells. Math.ceil on the size avoids sub-pixel gaps between adjacent rects.
    var cwCeil = Math.ceil(g.cw);
    var chCeil = Math.ceil(g.ch);
    for (r = 0; r < g.nRows; r++) {
        i = g.rowStart + r;
        y = MATRIX_LABEL_GUTTER + r * g.ch;
        for (c = 0; c < g.nCols; c++) {
            j = g.colStart + c;
            ctx.fillStyle = ColSch_getGradientColorString(values[i][j], minValue, maxValue);
            ctx.fillRect(MATRIX_LABEL_GUTTER + c * g.cw, y, cwCeil, chCeil);
            if (rowSel[r] && colSel[c]) {                   // interest-area intersection overlay
                ctx.fillStyle = 'rgba(250, 247, 228, 0.28)';
                ctx.fillRect(MATRIX_LABEL_GUTTER + c * g.cw, y, cwCeil, chCeil);
            }
        }
    }

    ctx.font = '9px sans-serif';

    // Column labels along the top gutter, rotated to read bottom-to-top (as the old table did).
    if (g.cw >= MATRIX_MIN_LABEL_CELL) {
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        for (c = 0; c < g.nCols; c++) {
            j = g.colStart + c;
            ctx.save();
            ctx.translate(MATRIX_LABEL_GUTTER + c * g.cw + g.cw / 2, MATRIX_LABEL_GUTTER - 3);
            ctx.rotate(-Math.PI / 2);
            ctx.fillStyle = colSel[c] ? '#2E2D26' : '#7e7b66';
            ctx.fillText(_fitLabel(ctx, '' + GVAR_pointsLabels[j], MATRIX_LABEL_GUTTER - 6), 0, 0);
            ctx.restore();
        }
    }

    // Row labels along the left gutter.
    if (g.ch >= MATRIX_MIN_LABEL_CELL) {
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (r = 0; r < g.nRows; r++) {
            i = g.rowStart + r;
            ctx.fillStyle = rowSel[r] ? '#2E2D26' : '#7e7b66';
            ctx.fillText(_fitLabel(ctx, '' + GVAR_pointsLabels[i], MATRIX_LABEL_GUTTER - 6),
                         MATRIX_LABEL_GUTTER - 4, MATRIX_LABEL_GUTTER + r * g.ch + g.ch / 2);
        }
    }

    // Edited-cell border, if it falls inside this quadrant.
    if (MATRIX_editedCell && MATRIX_editedCell.prefix === g.prefix &&
            MATRIX_editedCell.i >= g.rowStart && MATRIX_editedCell.i < g.rowEnd &&
            MATRIX_editedCell.j >= g.colStart && MATRIX_editedCell.j < g.colEnd) {
        ctx.strokeStyle = '#0047a4';
        ctx.lineWidth = 2;
        ctx.strokeRect(MATRIX_LABEL_GUTTER + (MATRIX_editedCell.j - g.colStart) * g.cw + 1,
                       MATRIX_LABEL_GUTTER + (MATRIX_editedCell.i - g.rowStart) * g.ch + 1,
                       g.cw - 2, g.ch - 2);
    }
}

/**
 * Redraw every currently visible quadrant canvas. This is the single render entry point,
 * used after edits, selection changes, quadrant/tab switches and resizes.
 */
function MATRIX_drawCanvas() {
    var canvases = document.getElementsByClassName('matrix-canvas');
    for (var k = 0; k < canvases.length; k++) {
        if (canvases[k].offsetWidth > 0) {                  // skip hidden quadrants
            _drawQuadrant(canvases[k]);
        }
    }
}

/**
 * Map a mouse event on a canvas to a matrix location.
 * @return null, or {region: 'cell'|'row'|'col', prefix, i, j, cellRect} where cellRect is in viewport coords.
 */
function _canvasHit(canvas, ev) {
    var g = _canvasGeometry(canvas);
    if (g.nCols <= 0 || g.nRows <= 0) {
        return null;
    }
    var rect = canvas.getBoundingClientRect();
    var x = ev.clientX - rect.left;
    var y = ev.clientY - rect.top;
    var inLeftGutter = x >= 0 && x < MATRIX_LABEL_GUTTER;
    var inTopGutter = y >= 0 && y < MATRIX_LABEL_GUTTER;
    var inGridX = x >= MATRIX_LABEL_GUTTER && x <= g.cssW;
    var inGridY = y >= MATRIX_LABEL_GUTTER && y <= g.cssH;
    var c = Math.floor((x - MATRIX_LABEL_GUTTER) / g.cw);
    var r = Math.floor((y - MATRIX_LABEL_GUTTER) / g.ch);
    var validC = c >= 0 && c < g.nCols;
    var validR = r >= 0 && r < g.nRows;

    if (inGridX && inGridY && validC && validR) {
        return {
            region: 'cell', prefix: g.prefix, i: g.rowStart + r, j: g.colStart + c,
            cellRect: {
                left: rect.left + MATRIX_LABEL_GUTTER + c * g.cw,
                top: rect.top + MATRIX_LABEL_GUTTER + r * g.ch,
                width: g.cw, height: g.ch
            }
        };
    }
    if (inLeftGutter && inGridY && validR) {                // left gutter -> row header
        return {region: 'row', prefix: g.prefix, i: g.rowStart + r, j: -1};
    }
    if (inTopGutter && inGridX && validC) {                 // top gutter -> column header
        return {region: 'col', prefix: g.prefix, i: -1, j: g.colStart + c};
    }
    return null;
}


/* ----------------------------------------------------------------------------------------
 * Tooltip shown while hovering a cell.
 * -------------------------------------------------------------------------------------- */

function _matrixTooltip() {
    var t = document.getElementById('matrixCellTooltip');
    if (!t) {
        t = document.createElement('div');
        t.id = 'matrixCellTooltip';
        t.style.cssText = 'position:fixed;display:none;pointer-events:none;z-index:10000;' +
            'background:rgba(46,45,38,0.92);color:#FAF7E4;font-size:11px;padding:3px 6px;' +
            'border-radius:3px;white-space:nowrap;';
        document.body.appendChild(t);
    }
    return t;
}

function _showTooltip(ev, prefix, i, j) {
    var value = _matrixByPrefix(prefix).values[i][j];
    var t = _matrixTooltip();
    t.textContent = GVAR_pointsLabels[i] + ' ← ' + GVAR_pointsLabels[j] + ' = ' + value;
    t.style.left = (ev.clientX + 14) + 'px';
    t.style.top = (ev.clientY + 12) + 'px';
    t.style.display = 'block';
}

function _hideTooltip() {
    var t = document.getElementById('matrixCellTooltip');
    if (t) {
        t.style.display = 'none';
    }
}


/* ----------------------------------------------------------------------------------------
 * Cell editing (context menu aside a cell).
 * -------------------------------------------------------------------------------------- */

/**
 * Position the edit menu next to the clicked cell, flipping when it would overflow the viewport.
 * The menu is fixed-positioned, so cellRect viewport coordinates are used directly.
 */
function _positionEditMenu(inputDiv, cellRect) {
    var menuWidth = 200;
    var menuHeight = 214;
    var pad = 6;

    var x = cellRect.left + cellRect.width + pad;
    if (x + menuWidth > $('body').width() - 15) {
        x = cellRect.left - menuWidth - pad;                // flip to the left
    }
    if (x < 0) {
        x = 5;
    }
    var y = cellRect.top;
    if (y + menuHeight > $(window).height()) {
        y = $(window).height() - menuHeight - 10;           // flip up
    }
    if (y < 0) {
        y = 5;
    }
    inputDiv.style.position = 'fixed';
    inputDiv.style.left = x + 'px';
    inputDiv.style.top = y + 'px';
}

/**
 * Open the edit menu for a single matrix cell.
 */
function changeSingleCell(hit) {
    var i = hit.i;
    var j = hit.j;
    if (!(GFUNC_isNodeAddedToInterestArea(i) && GFUNC_isNodeAddedToInterestArea(j))) {
        displayMessage("The node you selected is not in the current interest area!", "warningMessage");
    }

    MATRIX_editedCell = {prefix: hit.prefix, i: i, j: j};

    var inputDiv = document.getElementById('editNodeValues');
    inputDiv.style.display = 'block';
    _positionEditMenu(inputDiv, hit.cellRect);

    var labelInfoSource = document.getElementById('selectedSourceNodeDetails');
    var labelInfoTarget = document.getElementById('selectedTargetNodeDetails');
    if (labelInfoSource != null) {
        labelInfoSource.innerHTML = GVAR_pointsLabels[i];
    }
    if (labelInfoTarget != null) {
        labelInfoTarget.innerHTML = GVAR_pointsLabels[j];
    }

    document.getElementById('weightsValue').value = _matrixByPrefix(hit.prefix).values[i][j];
    // Keep the "td_<prefix>_<i>_<j>" identifier the edit helpers below parse indices from.
    document.getElementById('currentlyEditedNode').value = 'td_' + hit.prefix + '_' + i + '_' + j;

    MATRIX_drawCanvas();
}


/**
 * Method called when the 'Save' button from the context menu is pressed.
 * If a valid float is received store the value in the weights matrix and if not
 * display an error message. Either way close the details context menu.
 */
function saveNodeDetails() {
    var inputText = document.getElementById('weightsValue');
    var newValue = parseFloat($.trim(inputText.value));
    var indexes = _getIndexes();
    var idx = indexes[0];
    var jdx = indexes[1];

    if (isNaN(newValue)) {
        displayMessage('The value you entered is not a valid float. Original value is kept.', 'warningMessage');
    } else {
        var selectedMatrix = GVAR_interestAreaVariables[GVAR_selectedAreaType];

        if (newValue > selectedMatrix.max_val){
            selectedMatrix.max_val = newValue;
            CONN_initLinesHistorgram();
        }
        if (newValue < 0) {
            newValue = 0;
        }
        if (newValue < selectedMatrix.min_val){
            selectedMatrix.min_val = newValue;
            CONN_initLinesHistorgram();
        }
        if (selectedMatrix.values[idx][jdx] == selectedMatrix.max_val) {
            selectedMatrix.values[idx][jdx] = newValue;
            selectedMatrix.max_val = 0;
            for (var i=0; i<selectedMatrix.values.length; i++) {
                for (var j=0; j<selectedMatrix.values.length; j++) {
                    if (selectedMatrix.values[i][j] > selectedMatrix.max_val) {
                        selectedMatrix.max_val = selectedMatrix.values[i][j];
                    }
                }
            }
            CONN_initLinesHistorgram();
        }
        else {
            if (selectedMatrix.values[idx][jdx] == 0 && newValue > 0) {
                CONN_comingInLinesIndices[jdx].push(parseInt(idx));
                CONN_comingOutLinesIndices[idx].push(parseInt(jdx));
            }
            if (selectedMatrix.values[idx][jdx] > 0 && newValue == 0) {
                HLPR_removeByElement(CONN_comingInLinesIndices[jdx], parseInt(idx));
                HLPR_removeByElement(CONN_comingOutLinesIndices[idx], parseInt(jdx));
            }
            selectedMatrix.values[idx][jdx] = newValue;
            CONN_lineWidthsBins[idx][jdx] = CONN_getLineWidthValue(newValue);
        }
    }
    document.getElementById('editNodeValues').style.display = 'none';
    MATRIX_editedCell = null;

    MATRIX_colorTable();
    GFUNC_updateLeftSideVisualization();
}


/**
 * Hide the details context menu that pops up aside a edited element. This
 * method is called when pressing the 'Cancel' button or when clicking outside the table/canvas.
 */
function hideNodeDetails() {
    var hiddenNodeField = document.getElementById('currentlyEditedNode');
    if (hiddenNodeField.value != null && hiddenNodeField.value != "") {
        document.getElementById('editNodeValues').style.display = 'none';
        hiddenNodeField.value = null;
        MATRIX_editedCell = null;
        MATRIX_colorTable();
    }
}

function _getIndexes(){
    var prefix = GVAR_interestAreaVariables[GVAR_selectedAreaType].prefix;
    var hiddenNodeField = document.getElementById('currentlyEditedNode');
    return hiddenNodeField.value.split("td_" + prefix + '_')[1].split("_");
}

function _toggleCell(values, i, j){
    if (values[i][j] > 0) {
        if (GVAR_connectivityMatrix[i][j] === 1) {
            GVAR_connectivityMatrix[i][j] = 0;
        } else {
            GVAR_connectivityMatrix[i][j] = 1;
        }
    } else {
        GVAR_connectivityMatrix[i][j] = 0;
    }
}

/**
 * Method used to toggle between show/hide in-going lines. Used from the details context menu
 * aside a edited element.
 *
 * @param index - specified which of the two nodes is the one for which to make the toggle,
 *              0 = source node, 1 = destination node
 */
function toggleIngoingLines(index) {
    var values = GVAR_interestAreaVariables[GVAR_selectedAreaType].values;
    var idx = _getIndexes()[index];

    for (var i=0; i < NO_POSITIONS; i++) {
        _toggleCell(values, i, idx);
    }
    GFUNC_updateLeftSideVisualization();
}

/**
 * Method used to toggle between show/hide outgoing lines. Used from the details context menu
 * aside a edited element.
 *
 * @param index - specified which of the two nodes is the one for which to make the toggle,
 *        0 = source node, 1 = destination node
 */
function toggleOutgoingLines(index) {
    var values = GVAR_interestAreaVariables[GVAR_selectedAreaType].values;
    var idx = _getIndexes()[index];

    for (var i=0; i<NO_POSITIONS; i++) {
        _toggleCell(values, idx, i);
    }
    GFUNC_updateLeftSideVisualization();
}

/**
 * Method used to cut ingoing lines. Used from the details context menu
 * aside a edited element.
 *
 * @param index - specified which of the two nodes is the one for which to make the cut,
 *                0 = source node, 1 = destination node
 */
function cutIngoingLines(index) {
    var values = GVAR_interestAreaVariables[GVAR_selectedAreaType].values;
    var idx = _getIndexes()[index];
    var i;

    for (i=0; i<NO_POSITIONS; i++) {
        GVAR_connectivityMatrix[i][idx] = 0;
    }
    for (i=0; i<NO_POSITIONS; i++) {
        if (values[i][idx] > 0){
            HLPR_removeByElement(CONN_comingInLinesIndices[idx], parseInt(i));
            HLPR_removeByElement(CONN_comingOutLinesIndices[i], parseInt(idx));
        }
        values[i][idx] = 0;
    }
    MATRIX_colorTable();
    GFUNC_updateLeftSideVisualization();
}

/**
 * Method used to cut outgoing lines. Used from the details context menu
 * aside a edited element.
 *
 * @param index - specified which of the two nodes is the one for which to make the cut,
 *                0 = source node, 1 = destination node
 */
function cutOutgoingLines(index) {
    var values = GVAR_interestAreaVariables[GVAR_selectedAreaType].values;
    var idx = _getIndexes()[index];

    for (var i=0; i<NO_POSITIONS; i++) {
        if (values[idx][i] > 0){
            HLPR_removeByElement(CONN_comingInLinesIndices[i], parseInt(idx));
            HLPR_removeByElement(CONN_comingOutLinesIndices[idx], parseInt(i));
        }
        GVAR_connectivityMatrix[idx][i] = 0;
        values[idx][i] = 0;
    }
    MATRIX_colorTable();
    GFUNC_updateLeftSideVisualization();
}


/* ----------------------------------------------------------------------------------------
 * Interest-area selection from row/column headers.
 * -------------------------------------------------------------------------------------- */

function _toggleNode(index){
    GFUNC_toggleNodeInInterestArea(index);
    // The selection comp will trigger a change event. We subscribe to that and do a bulk redraw.
    GFUN_updateSelectionComponent();
}

/**
 * Toggle the whole column (node j) in/out of the interest area. Called on a top-gutter click.
 */
function changeEntireColumn(index) {
    _toggleNode(index);
}

/**
 * Toggle the whole row (node i) in/out of the interest area. Called on a left-gutter click.
 */
function changeEntireRow(index) {
    _toggleNode(index);
}

/**
 * Redraw the matrix to reflect the current interest-area selection.
 */
function refreshTableInterestArea() {
    MATRIX_drawCanvas();
}


/* ----------------------------------------------------------------------------------------
 * Legend and full redraw.
 * -------------------------------------------------------------------------------------- */

/**
 * Function to update the legend colors; the gradient is sized to the drawn matrix.
 * @private
 */
function _updateLegendColors(){
    var selectedMatrix = GVAR_interestAreaVariables[GVAR_selectedAreaType];
    var div_id = selectedMatrix.legend_div_id;
    var legendDiv = document.getElementById(div_id);

    var height = Math.max($("#div-matrix-weights")[0].clientHeight, $("#div-matrix-tracts")[0].clientHeight);
    ColSch_updateLegendColors(legendDiv, height);

    ColSch_updateLegendLabels('#table-' + div_id, selectedMatrix.min_val, selectedMatrix.max_val, height);
}


/**
 * Redraw the matrix and its legend. Kept under the historical name because it is the
 * hook called from the color scheme, tab-switch and selection code.
 */
function MATRIX_colorTable() {
    MATRIX_drawCanvas();
    _updateLegendColors();
}

function saveSubConnectivity(submitUrl, originalConnectivityId,  isBranch) {
    var data = {
        original_connectivity: originalConnectivityId,
        new_weights: $.toJSON(GVAR_interestAreaVariables[1].values),
        new_tracts: $.toJSON(GVAR_interestAreaVariables[2].values),
        interest_area_indexes: $.toJSON(GVAR_interestAreaNodeIndexes),
        User_Tag_1_Perpetuated: $('#newConnectivityNameTag').val()
    };
    // Emulate the way browsers send checkboxes in forms.
    // They send a value only for the checked ones. All values are true
    if (isBranch){
        data.is_branch = 'True';
    }

    doAjaxCall({ url: submitUrl, data: data });
    displayMessage("Launched connectivity creator", "importantMessage");
}

/**
 * Bind events for the connectivity matrix canvases.
 */
function initializeMatrix(){
    _matrixTooltip();

    function onClick(ev){
        var hit = _canvasHit(this, ev);
        if (hit == null){
            return;
        }
        if (hit.region === 'cell'){
            changeSingleCell(hit);
        } else if (hit.region === 'row'){
            changeEntireRow(hit.i);
        } else if (hit.region === 'col'){
            changeEntireColumn(hit.j);
        }
    }

    function onMove(ev){
        var hit = _canvasHit(this, ev);
        if (hit != null && hit.region === 'cell'){
            highlightedPointIndex1 = hit.i;
            highlightedPointIndex2 = hit.j;
            _showTooltip(ev, hit.prefix, hit.i, hit.j);
        } else {
            highlightedPointIndex1 = -1;
            highlightedPointIndex2 = -1;
            _hideTooltip();
        }
    }

    function onLeave(){
        highlightedPointIndex1 = -1;
        highlightedPointIndex2 = -1;
        _hideTooltip();
    }

    var canvases = document.getElementsByClassName('matrix-canvas');
    for (var k = 0; k < canvases.length; k++){
        canvases[k].addEventListener('click', onClick);
        canvases[k].addEventListener('mousemove', onMove);
        canvases[k].addEventListener('mouseleave', onLeave);
    }

    // Redraw on resize so cells keep filling the available width. Throttled to avoid thrashing.
    var resizeTimer;
    $(window).resize(function(){
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(MATRIX_drawCanvas, 200);
    });

    MATRIX_colorTable();
}
