var Quadrant = function (params){                // this keeps all necessary data for drawing
    this.index = params.index || 0;                // in a quadrant
    this.axes = params.axes || {x: 0, y: 1};       // axes represented in current quad; i=0, j=1, k=2
    this.entityWidth = params.entityWidth || 0;    // width and height of one voxel in current quad
    this.entityHeight = params.entityHeight || 0;
    this.offsetX = params.offsetX || 0;            // the offset of the drawing relative to the quad
    this.offsetY = params.offsetY || 0;
};


/**
 * Draws the selectedQuadrant on Focus Quadrant from the xyz planes data.
 */
function drawFocusQuadrantFromView(sliceArray){
    _setCtxOnQuadrant(3);
    if(tsVol.highlightedQuad.index === 0){
        for (var j = 0; j < tsVol.dataSize[2]; ++j){
            for (var i = 0; i < tsVol.dataSize[1]; ++i){
                drawVoxel(i, j, sliceArray[0][i][j]);
            }
        }
    } else if(tsVol.highlightedQuad.index === 1){
        for (var k = 0; k < tsVol.dataSize[3]; ++k){
            for (var jj = 0; jj < tsVol.dataSize[2]; ++jj){
                drawVoxel(k, jj, sliceArray[1][jj][k]);
            }
        }
    } else if(tsVol.highlightedQuad.index === 2){
        for (var kk = 0; kk < tsVol.dataSize[3]; ++kk){
            for (var ii = 0; ii < tsVol.dataSize[1]; ++ii){
                drawVoxel(kk, ii, sliceArray[2][ii][kk]);
            }
        }
    }
    drawMargin();
}

/**
 * Draws the voxel set at (line, col) in the current quadrant, and colors it
 * according to its value.
 * This function know nothing about the time point.
 * @param line THe vertical line of the grid we wish to draw on
 * @param col The horizontal line of the grid we wish to draw on
 * @param value The value of the voxel that will be converted into color
 */
function drawVoxel(line, col, value){
    tsVol.ctx.fillStyle = ColSch_getAbsoluteGradientColorString(value);
    // col increases horizontally and line vertically, so col represents the X drawing axis, and line the Y
	tsVol.ctx.fillRect(col * tsVol.currentQuadrant.entityWidth, line * tsVol.currentQuadrant.entityHeight,
                       tsVol.currentQuadrant.entityWidth + 1, tsVol.currentQuadrant.entityHeight + 1);
}

/**
 * Draws the cross-hair on each quadrant, on the <code>tsVol.selectedEntity</code>
 */
function drawNavigator(){
    // Preview quadrans navigators
    tsVol.ctx.save();
    tsVol.ctx.beginPath();

    for (var quadIdx = 0; quadIdx < 3; ++quadIdx){
        _setCtxOnQuadrant(quadIdx);
        var x = tsVol.selectedEntity[tsVol.currentQuadrant.axes.x] * tsVol.currentQuadrant.entityWidth + tsVol.currentQuadrant.entityWidth / 2;
        var y = tsVol.selectedEntity[tsVol.currentQuadrant.axes.y] * tsVol.currentQuadrant.entityHeight + tsVol.currentQuadrant.entityHeight / 2;
        drawCrossHair(x, y);
    }
    tsVol.ctx.strokeStyle = "red";
    tsVol.ctx.lineWidth = 3;
    tsVol.ctx.stroke();
    tsVol.ctx.restore();

    // Focus quadrant Navigator
    tsVol.ctx.save();
    tsVol.ctx.beginPath();

    _setCtxOnQuadrant(3);
    var xx = tsVol.selectedEntity[tsVol.currentQuadrant.axes.x] * tsVol.currentQuadrant.entityWidth + tsVol.currentQuadrant.entityWidth / 2;
    var yy = tsVol.selectedEntity[tsVol.currentQuadrant.axes.y] * tsVol.currentQuadrant.entityHeight + tsVol.currentQuadrant.entityHeight / 2;
    drawFocusCrossHair(xx, yy);

    tsVol.ctx.strokeStyle = "blue";
    tsVol.ctx.lineWidth = 3;
    tsVol.ctx.stroke();
    tsVol.ctx.restore();
}

/**
 * Draws a 20px X 20px cross hair on the <code>tsVol.currentQuadrant</code>, at the specified x and y
 */
function drawCrossHair(x, y){
    tsVol.ctx.moveTo(Math.max(x - 20, 0), y);                              // the horizontal line
    tsVol.ctx.lineTo(Math.min(x + 20, tsVol.quadrantWidth), y);
    tsVol.ctx.moveTo(x, Math.max(y - 20, 0));                              // the vertical line
    tsVol.ctx.lineTo(x, Math.min(y + 20, tsVol.quadrantHeight));
}

/**
 * Draws a cross hair on the bigger focus quadrant, at the specified x and y
 */
function drawFocusCrossHair(x, y){
    tsVol.ctx.moveTo(Math.max(x - 20, 0), y);                              // the horizontal line
    tsVol.ctx.lineTo(Math.min(x + 20, tsVol.focusQuadrantWidth), y);
    tsVol.ctx.moveTo(x, Math.max(y - 20, 0));                              // the vertical line
    tsVol.ctx.lineTo(x, Math.min(y + 20, tsVol.focusQuadrantHeight));
}

/**
 * Draws a canvas legend at the bottom of the volume visualizer:
 * <ul>
 * <li>Written min, mean and max values for the data, in scientific notation.
 * <li>Displays the color range for the data.
 * <li>Displays a white bar in to show the currently selected entity value.
 * </ul>
 */
function drawLegend(selectedEntityValue) {
    var tmp = tsVol.legendPadding / 2;
    // set the context on the legend quadrant
    tsVol.ctx.setTransform(1, 0, 0, 1, 0, 0);  // reset the transformation
    tsVol.ctx.translate( tsVol.quadrantWidth + tmp, tsVol.focusQuadrantHeight);
    // set the font properties
    tsVol.ctx.font = '12px Helvetica';
    tsVol.ctx.textAlign = 'center';
    tsVol.ctx.textBaseline = 'middle';
    tsVol.ctx.fillStyle = 'white';
    // write min, mean, max values on the canvas
    tsVol.ctx.fillText(tsVol.minimumValue.toExponential(2), 0, tsVol.legendHeight - 10);
    tsVol.ctx.fillText("|", 1, tsVol.legendHeight/2);
    var midValue = (tsVol.maximumValue - tsVol.minimumValue)/2;
    tsVol.ctx.fillText(midValue.toExponential(2), tsVol.legendWidth/2, tsVol.legendHeight - 10);
    tsVol.ctx.fillText("|", tsVol.legendWidth/2, tsVol.legendHeight/2);
    tsVol.ctx.fillText(tsVol.maximumValue.toExponential(2), tsVol.legendWidth, tsVol.legendHeight - 10);
    tsVol.ctx.fillText("|", tsVol.legendWidth-1, tsVol.legendHeight/2);

    // Draw a color bar from min to max value based on the selected color coding
    for(var i = 0; i< tsVol.legendWidth; i++){
        var val = tsVol.minimumValue + ((i/tsVol.legendWidth)*(tsVol.maximumValue-tsVol.minimumValue));
        tsVol.ctx.fillStyle = ColSch_getAbsoluteGradientColorString(val);
        tsVol.ctx.fillRect(i, 1, 1.5, tsVol.legendHeight/2);
    }

    // Draw the selected entity value marker on the color bar
    tsVol.ctx.fillStyle = "white";
    tmp = (selectedEntityValue - tsVol.minimumValue) / (tsVol.maximumValue - tsVol.minimumValue);
    tsVol.ctx.fillRect(tmp * tsVol.legendWidth, 1, 3, tsVol.legendHeight/2);
}

/**
 * Add spatial labels to the navigation quadrants
 */
function drawLabels(){
    tsVol.ctx.font = '15px Helvetica';
    tsVol.ctx.textBaseline = 'middle';
    _setCtxOnQuadrant(0);
    tsVol.ctx.fillText("Axial", tsVol.quadrantWidth/5, tsVol.quadrantHeight - 13);
    _setCtxOnQuadrant(1);
    tsVol.ctx.fillText("Sagittal", tsVol.quadrantWidth/5, tsVol.quadrantHeight - 13);
    _setCtxOnQuadrant(2);
    tsVol.ctx.fillText("Coronal", tsVol.quadrantWidth/5, tsVol.quadrantHeight - 13);
}

/**
 * Draws a 5px rectangle around the <code>tsVol.currentQuadrant</code>
 */
function drawMargin(){
    var marginWidth, marginHeight;
    if(tsVol.currentQuadrant.index === 3){
        marginWidth = tsVol.focusQuadrantWidth;
        marginHeight = tsVol.focusQuadrantHeight;
    }
    else{
        marginWidth = tsVol.quadrantWidth;
        marginHeight = tsVol.quadrantHeight;
    }
    tsVol.ctx.beginPath();
    tsVol.ctx.rect(2, 0, marginWidth - 3, marginHeight - 2);
    tsVol.ctx.lineWidth = 2;
    if(tsVol.currentQuadrant.index === tsVol.selectedQuad.index && tsVol.currentQuadrant.index !== 3){
        tsVol.ctx.strokeStyle = 'white';
        tsVol.highlightedQuad = tsVol.currentQuadrant;
    }
    else if(tsVol.currentQuadrant.index === tsVol.highlightedQuad.index && tsVol.selectedQuad.index === 3){
        tsVol.ctx.strokeStyle = 'white';
    }
    else{
        tsVol.ctx.strokeStyle = 'gray';
    }
    tsVol.ctx.stroke();
}
