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

/* globals  GVAR_connectivityMatrix, GVAR_interestAreaVariables, GVAR_selectedAreaType,
            GVAR_pointsLabels, GVAR_interestAreaNodeIndexes, HLPR_removeByElement,
            GFUNC_updateLeftSideVisualization, GFUNC_isNodeAddedToInterestArea,
            GFUN_updateSelectionComponent, GFUNC_toggleNodeInInterestArea,
            displayMessage,
            CONN_getLineWidthValue, CONN_lineWidthsBins, CONN_comingInLinesIndices, CONN_comingOutLinesIndices, CONN_initLinesHistorgram,
            NO_POSITIONS, highlightedPointIndex1, highlightedPointIndex2,
            ColSch_updateLegendColors, ColSch_updateLegendLabels
*/

/*
 * This file handles the display and functionality of the 2d table view
 */
/*
 * Used to keep track of the start and end point for each quadrant.
 */
var startPointsX = [];
var endPointsX = [];
var startPointsY = [];
var endPointsY = [];

/*
 * Keep references to the last edited element, element color, and element class.
 * To be used for switching back to the original after an edit is performed.
 */
var lastEditedElement = null;
var lastElementColor = null;
var lastElementClass = null;

/**
 * Get the position of a element in the page. Used when finding out the position where the 
 * menu with information about a specific edge should be displayed.
 * 
 * @param elem - the element for which you want the absolute position in the page
 * 
 * @return {x: Number, y: Number} - a dictionary with the offset numbers
 */
function getMenuPosition(elem, contextMenuDiv){  
   
    var posX = 210;  // Default offset
    var posY = 15;

    while(elem != null){
        posX += elem.offsetLeft;
        posY += elem.offsetTop;
        elem = elem.offsetParent;
    }
    var $w = $("#scrollable-matrix-section");
    posY -= $w.scrollTop();
    if ($w[0].offsetTop > 0) {
        posY -= $w[0].offsetTop;
        posY -= ($("#main").scrollTop() - $w[0].offsetTop);
    }
    //posX -= $w.scrollLeft()

    var mh = 214; //$(contextMenuDiv).height();
    //var mw = 200; //$(contextMenuDiv).width()
    var ww = $("body").width() - 15;
    var wh = Math.max($(window).height(), $w.height());

    var maxRight = posX;
    if (maxRight > ww) {
        posX -= (maxRight - ww);
    }

    var dir = "down";
    if (posY + mh > wh) {
        dir = "up";
    }
    if (dir == "up") {
        posY -= (mh + 25);
    }
    return {x : posX, y : posY };
} 

/**
 * Method called on the click event of a table box that represents a certain node from the connectivity matrix
 *
 * @param table_elem the dom element which fired the click event
 * @param i
 * @param j
 */
function changeSingleCell(table_elem, i, j) {

    var inputDiv = document.getElementById('editNodeValues');
    if (!(GFUNC_isNodeAddedToInterestArea(i) && GFUNC_isNodeAddedToInterestArea(j))) {
        displayMessage("The node you selected is not in the current interest area!", "warningMessage");
    }
    if (inputDiv.style.display == 'none') {
        inputDiv.style.display = 'block';
    } else {
        lastEditedElement.className = lastElementClass;
    }
    lastEditedElement = table_elem;
    lastElementClass = table_elem.className;
    table_elem.className = "edited";
    var element_position = getMenuPosition(table_elem, inputDiv);
    inputDiv.style.position = 'fixed';
    inputDiv.style.left = element_position.x + 'px';
    inputDiv.style.top = element_position.y + 'px';

    var labelInfoSource = document.getElementById('selectedSourceNodeDetails');
    var labelInfoTarget = document.getElementById('selectedTargetNodeDetails');
    var descriptionText = GVAR_pointsLabels[i];
    if (labelInfoSource != null) {
        labelInfoSource.innerHTML = descriptionText;
    }
    descriptionText = GVAR_pointsLabels[j];
    if (labelInfoTarget != null) {
        labelInfoTarget.innerHTML = descriptionText;
    }

    var inputText = document.getElementById('weightsValue');
    inputText.value = GVAR_interestAreaVariables[GVAR_selectedAreaType]['values'][i][j];

    var hiddenNodeField = document.getElementById('currentlyEditedNode');
    hiddenNodeField.value = table_elem.id;
    MATRIX_colorTable();
}


/**
 * Method called when the 'Save' button from the context menu is pressed.
 * If a valid float is recieverm store the value in the weights matrix and if not
 * display an error message. Either way close the details context menu.
 */
function saveNodeDetails() {
    var inputText = document.getElementById('weightsValue');
    var newValue = parseFloat($.trim(inputText.value));
    var hiddenNodeField = document.getElementById('currentlyEditedNode');
    var tableNodeID = hiddenNodeField.value;
    var table_element = document.getElementById(tableNodeID);
    table_element.className = lastElementClass;

    if (isNaN(newValue)) {
        displayMessage('The value you entered is not a valid float. Original value is kept.', 'warningMessage');
    } else {
        //displayMessage('')
        var selectedMatrix = GVAR_interestAreaVariables[GVAR_selectedAreaType];
        var indexes = tableNodeID.split("td_" + selectedMatrix.prefix + '_')[1].split("_");
        var idx = indexes[0];
        var jdx = indexes[1];

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
    var inputDiv = document.getElementById('editNodeValues');
    inputDiv.style.display = 'none';
    lastElementClass = null;
    lastEditedElement = null;
    lastElementColor = null;

    MATRIX_colorTable();
    GFUNC_updateLeftSideVisualization();
}


/**
 * Hide the details context menu that pops up aside a edited element. This
 * method is called when pressing the 'Cancel' button or when clicking outside the table/canvas.
 */
function hideNodeDetails() {
    var inputDiv = document.getElementById('editNodeValues');
    var hiddenNodeField = document.getElementById('currentlyEditedNode');
    var tableNodeID = hiddenNodeField.value;
    if (tableNodeID != null && tableNodeID != "") {
        inputDiv.style.display = 'none';
        if (lastEditedElement != null) {
            lastEditedElement.className = lastElementClass;
            lastEditedElement.style.backgroundColor = lastElementColor;
        }
        hiddenNodeField.value = null;
        lastElementClass = null;
        lastEditedElement = null;
        lastElementColor = null;
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



function refreshTableInterestArea() {
    if ($('#div-matrix-tracts').length > 0) {  // why this check?
        for (var i = 0; i < NO_POSITIONS; i++) {
            _updateNodeInterest(i);
        }
    }
}

/**
 * Efficiently get header buttons by constructing their id's instead of searching the dom
 * This replaces $("th[id^='upper_change_" + nodeIdx + "_']"); $("td[id^='left_change_" + nodeIdx + "_']");
 */
function _get_header_buttons(nodeIdx){
    function addExistingEl(list, id){
        var el = document.getElementById(id);
        if (el != null){
            list.push(el);
        }
    }

    var hemisphereSuffixes = ['leftHemisphere', 'leftRightQuarter', 'rightLeftQuarter', 'rightHemisphere'];
    var upperSideButtons = [];
    var leftSideButtons = [];

    for (var i = 0; i < hemisphereSuffixes.length; i++){
        var wuBtnId = 'upper_change_' + nodeIdx + '_' + hemisphereSuffixes[i];
        var wlBtnId = 'left_change_' + nodeIdx + '_' + hemisphereSuffixes[i];

        addExistingEl(upperSideButtons, wuBtnId);
        addExistingEl(leftSideButtons, wlBtnId);
        addExistingEl(upperSideButtons, wuBtnId + 'Tracts');
        addExistingEl(leftSideButtons, wlBtnId + 'Tracts');
    }

    return {'u': upperSideButtons, 'l': leftSideButtons};
}

/**
 * For a given node index update the style of the table correspondingly.
 * This is function is now indended for bulk table updates.
 * @private used by refreshTableInterestArea
 */
function _updateNodeInterest(nodeIdx) {
    var isInInterest = GFUNC_isNodeAddedToInterestArea(nodeIdx);
    var hb = _get_header_buttons(nodeIdx);
    var upperSideButtons = hb.u;
    var leftSideButtons = hb.l;

    var prefix = GVAR_interestAreaVariables[GVAR_selectedAreaType].prefix;
    var k;

    for (k = 0; k < upperSideButtons.length; k++) {
        if (isInInterest) {
            upperSideButtons[k].className = 'selected';
        } else {
            upperSideButtons[k].className = '';
        }
    }
    
    for (k = 0; k < leftSideButtons.length; k++) {
        if (isInInterest) {
            leftSideButtons[k].className = 'identifier selected';
        } else {
            leftSideButtons[k].className = 'identifier';
        }
    }    
    
    for (var i = 0; i < NO_POSITIONS; i++){
        var horiz_table_data_id = 'td_' + prefix + '_' + nodeIdx + '_' + i;
        var horiz_table_element = document.getElementById(horiz_table_data_id);

        if (isInInterest && GFUNC_isNodeAddedToInterestArea(i)) {
            horiz_table_element.className = 'selected';
        } else {
            horiz_table_element.className = '';
        }
    }
}

function _toggleNode(index){
    GFUNC_toggleNodeInInterestArea(index);
    // The selection comp will trigger a change event. We subscribe to that and do a bulk table update
    GFUN_updateSelectionComponent();
}
/**
 * Method called when clicking on a node index from the top column. Change the entire column
 * associated with that index
 *
 * @param domElem the dom element which fired the click event
 */  
function changeEntireColumn(domElem) {
    var index = domElem.id.split("upper_change_")[1];
    index = parseInt(index.split('_')[0]);
    _toggleNode(index);
}


/**
 * Method called when clicking on a node label from the left column. Change the entire row
 * associated with that index
 *
 * @param domElem the dom element which fired the click event
 */  
function changeEntireRow(domElem) {
    var index = domElem.id.split("left_change_")[1];
    index = parseInt(index.split('_')[0]);
    _toggleNode(index);
}


/**
 * Helper methods that store information used when the colorTable method is called
 */

function TBL_storeHemisphereDetails(newStartPointsX, newEndPointsX, newStartPointsY, newEndPointsY) {
    startPointsX = eval(newStartPointsX);
    endPointsX = eval(newEndPointsX);
    startPointsY = eval(newStartPointsY);
    endPointsY = eval(newEndPointsY);
}

/**
 * Function to update the legend colors; the gradient will be created only after the table was drawn
 * so it will have the same size as the table matrix
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
 * Method that colors the entire table.
 */
function MATRIX_colorTable() {
    var selectedMatrix = GVAR_interestAreaVariables[GVAR_selectedAreaType];
    var prefix_id = selectedMatrix.prefix;
    var dataValues = selectedMatrix.values;
    var minValue = selectedMatrix.min_val;
    var maxValue = selectedMatrix.max_val;

    for (var hemisphereIdx=0; hemisphereIdx<startPointsX.length; hemisphereIdx++){
        var startX = startPointsX[hemisphereIdx];
        var endX = endPointsX[hemisphereIdx];
        var startY = startPointsY[hemisphereIdx];
        var endY = endPointsY[hemisphereIdx];

        for (var i=startX; i<endX; i++){
            for (var j=startY; j<endY; j++) {
                var tableDataID = 'td_' + prefix_id + '_' + i + '_' + j;
                var tableElement = document.getElementById(tableDataID);
                if (dataValues){
                    tableElement.style.backgroundColor = ColSch_getGradientColorString(dataValues[i][j], minValue, maxValue);
                }
            }
        }
    }
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
 * Bind events for connectivity matrix tables
 */
function initializeMatrix(){
    function tdInfo(el){
        var sid = el.id.split('_');
        var isNode = el.tagName === 'TD' && sid[1] != null && sid[2] != null && sid[3] != null;
        var prefix = sid[1],
                 i = parseInt(sid[2], 10),
                 j = parseInt(sid[3], 10);
        return { prefix : prefix, i : i, j : j, isNode : isNode};
    }

    function handle_click(el){
        var nfo = tdInfo(el);
        if (nfo.isNode){
            changeSingleCell(el, nfo.i, nfo.j);
        }
    }

    var dom = $('#div-matrix-weights').add('#div-matrix-tracts');
    dom.click(function(ev){
        handle_click(ev.target);
    }).keypress(function(ev){
        if (ev.keyCode === 13){
            handle_click(ev.target);
        }
    });

    // this binds quite a number of handlers
    dom.find('td').hover(
        function (event) {
            var nfo = tdInfo(event.target);
            if (nfo.isNode){
                highlightedPointIndex1 = nfo.i;
                highlightedPointIndex2 = nfo.j;
            }
        },
        function () {
            highlightedPointIndex1 = -1;
            highlightedPointIndex2 = -1;
        }
    );
}