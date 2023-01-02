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
 * Functions for BrainVisualizer in Double View (eeg-lines + 3D activity).
 */

function _EX_commonChannelInit(updateColorBufferForMeasurePoint){
    AG_regionSelector.change(function(selection){
           // update brain region filter
            VS_selectedRegions = AG_submitableSelectedChannels.slice();

            // update the measure point colors
            for(i=0; i != AG_regionSelector._allValues.length; i++){
                var val = AG_regionSelector._allValues[i];
                var selected = selection.indexOf(val) != -1;
                val = parseInt(val, 10);
                updateColorBufferForMeasurePoint(val, selected);
            }
            initActivityData();
        }
    );

    // subscribe the 3d view to mode and state var changes
    AG_modeSelector.modeChanged(VS_changeMode);
    AG_modeSelector.stateVariableChanged(VS_changeStateVariable);

    // initialize brain region filter with animated graph selection
    var selection = AG_regionSelector.val();
    for(var i=0; i < selection.length; i++){
        var val = parseInt(selection[i], 10);
        VS_selectedRegions.push(val);
        // assuming previous brain selection was void update selected buffers
        updateColorBufferForMeasurePoint(val, true);
    }

    // "subscribe" to measure point selection.
    // For consistency with other 3d connectivity views the brain sets this global variable when a measure point is selected.
    $('#GLcanvas').click(function(){
        if (VS_pickedIndex != null && VS_pickedIndex !== -1){
            _EX_onPickedMeasurePoint(VS_pickedIndex);
        }
    });
}

function EX_initializeChannels() {
    _EX_commonChannelInit(EX_changeColorBufferForMeasurePoint);
}

function _EX_onPickedMeasurePoint(measurePointIndex){
    // assumes that the index is the same with the value of the checkboxes
    var idx = AG_submitableSelectedChannels.indexOf(measurePointIndex);
    if(idx != -1){
        AG_submitableSelectedChannels.splice(idx, 1);
    }else{
        AG_submitableSelectedChannels.push(measurePointIndex);
    }
    AG_regionSelector.val(AG_submitableSelectedChannels);
}

/**
 * In the extended view if a certain EEG channel was selected then we
 * have to draw the measure point corresponding to it with a different color.
 *
 * @param measurePointIndex the index of the measure point to which correspond the EEG channel
 * @param isPicked if <code>true</code> then the point will be drawn with the color corresponding
 * to the selected channels, otherwise with the default color
 */
function EX_changeColorBufferForMeasurePoint(measurePointIndex, isPicked) {
    var colorBufferIndex = measurePointsBuffers[measurePointIndex].length - 1;
    measurePointsBuffers[measurePointIndex][colorBufferIndex] = createColorBufferForCube(isPicked);
}

/**
 * Initialization function for Channels, when sensor internals.
 */
function EX_initializeChannelsForSensorsInternal() {
    _EX_commonChannelInit(_changeColorBufferForMeasurePointSensorInternal);
}

/**
 * In the extended view if a certain EEG channel was selected then we
 * have to draw the measure point corresponding to it with a different color.
 *
 * @param measurePointIndex the index of the measure point to which correspond the EEG channel
 * @param isPicked if <code>true</code> then the point will be drawn with the color corresponding
 * to the selected channels, otherwise with the default color
 */
function _changeColorBufferForMeasurePointSensorInternal(measurePointIndex, isPicked) {
    var colorBufferIndex = measurePointsBuffers[measurePointIndex].length - 1;
    var vertexRegionBuffer = VSI_createColorBufferForSphere(measurePointIndex, measurePointsBuffers[measurePointIndex][0].numItems * 3);
    measurePointsBuffers[measurePointIndex][colorBufferIndex] = vertexRegionBuffer;
}
