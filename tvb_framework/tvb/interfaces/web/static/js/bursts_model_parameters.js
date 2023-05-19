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

/* globals TVBUI, doAjaxCall, displayMessage */

var modelParam = {
    dynamics: {}, // a constant collection of available dynamics indexed by id
    modelsInNodes: [] // a list of dynamics placed in nodes. This is the data model of the viewer.
};


(function(TVBUI, doAjaxCall, displayMessage){

function putModelInSelectedNodes(selected){
    var dynamic_id = $('#current_dynamic').val();
    var dynamic = modelParam.dynamics[dynamic_id];

    for (var i = 0; i < selected.length; i++) {
        var nodeId = selected[i];
        modelParam.modelsInNodes[nodeId] = dynamic;
    }
    return dynamic.name;
}

function prepareSubmitData(){
    var dyn_ids = [];
    var first = modelParam.modelsInNodes[0];

    for (var i = 0; i < modelParam.modelsInNodes.length; i++){
        var dyn = modelParam.modelsInNodes[i];
        if (dyn == null){
            displayMessage("node " + i + " is empty", "warningMessage");
            return null;
        }
        if (dyn.model_class !== first.model_class){
            displayMessage("all nodes must contain the same model type", "warningMessage");
            return null;
        }
        dyn_ids.push(dyn.id);
    }
    return dyn_ids;
}

function onShowDynamicDetails(){
    var dynamic_id = $('#current_dynamic').val();
    doAjaxCall({
        url:'/burst/dynamic/dynamic_detail/' + dynamic_id,
        success:function(frag){
            $('#dynamic-detail').find('.dropdown-pane').html(frag);
        }
    });
}

function _setInitialDynamics(initialDynamicIds){
    var texts = [];

    for(var i = 0; i < initialDynamicIds.length; i++){
        var dynamic_id = initialDynamicIds[i];
        var dynamic = modelParam.dynamics[dynamic_id];
        modelParam.modelsInNodes[i] = dynamic;
        texts.push(dynamic.name);
    }
    modelParam.view.setGridText(texts);
}

function main(dynamics, initialDynamicIds, selectionGID){
    modelParam.view = new TVBUI.RegionAssociatorView({
        selectionGID: selectionGID,
        onPut: putModelInSelectedNodes,
        prepareSubmitData: prepareSubmitData
    });

    // it might be better to send the node labels array to main
    var number_of_nodes = modelParam.view.selector._allValues.length;

    for(var i = 0; i < number_of_nodes; i++){
        modelParam.modelsInNodes[i] = null;
    }

    modelParam.dynamics = dynamics;
    _setInitialDynamics(initialDynamicIds);

    $('#dynamic-detail').find('button').click(onShowDynamicDetails);
}

modelParam.main = main;

})(TVBUI, doAjaxCall, displayMessage);


