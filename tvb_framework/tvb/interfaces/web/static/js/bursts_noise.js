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
 * .. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
 **/

/* globals TVBUI */

var noiseParam = {
    stateVars: [],
    noiseInNodes : []
};


(function(TVBUI){
    
function readNoiseInputs(){
    var ret = {};
    for (var i = 0; i < noiseParam.stateVars.length; i++){
        var sv = noiseParam.stateVars[i];
        ret[sv] = parseFloat($('#noisevalue_' + i).val());
    }
    return ret;
}

function formatNoiseDict(dispersions){
    var ret = [];
    for (var i = 0; i < noiseParam.stateVars.length; i++){
        var sv = noiseParam.stateVars[i];
        ret.push(dispersions[sv]);
    }
    return '[ ' + ret.join(',  ') + ' ]';
}

function putNoiseInSelectedNodes(selected){
    var dispersions = readNoiseInputs();

    for (var i = 0; i < selected.length; i++) {
        var nodeId = selected[i];
        noiseParam.noiseInNodes[nodeId] = dispersions;
    }
    return formatNoiseDict(dispersions);
}

function _setInitialNoise(initialNoise){
    var texts = [];
    for(var i = 0; i < initialNoise.length; i++){
        var dispersions = initialNoise[i];
        noiseParam.noiseInNodes[i] = dispersions;
        texts.push(formatNoiseDict(dispersions));
    }
    noiseParam.view.setGridText(texts);
}

function main(stateVars, initialNoise, selectionGID){
    noiseParam.view = new TVBUI.RegionAssociatorView({
        selectionGID: selectionGID,
        onPut: putNoiseInSelectedNodes,
        prepareSubmitData: function(){ return noiseParam.noiseInNodes; }
    });

    noiseParam.stateVars = stateVars;
    _setInitialNoise(initialNoise);
}

noiseParam.main = main;

})(TVBUI);


