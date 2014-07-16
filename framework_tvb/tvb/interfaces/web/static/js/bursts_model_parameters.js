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
 * http://www.gnu.org/licenses/old-licenses/gpl-2.
 *
 *   CITATION:
 * When using The Virtual Brain for scientific publications, please cite it as follows:
 *
 *   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
 *   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
 *       The Virtual Brain: a simulator of primate brain network dynamics.
 *   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
 *
 * .. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
 **/

/* globals doAjaxCall, displayMessage, TVBUI, GVAR_interestAreaNodeIndexes*/


var modelParam = {
    selector: null,
    dynamics: {}, // a constant collection of available dynamics indexed by id
    modelsInNodes: [] // a list of dynamics placed in nodes. This is the data model of the viewer.
};


(function(){

function initSelector(selectionGID){
    var selector = TVBUI.textGridRegionSelector("#channelSelector", {filterGid: selectionGID, emptyValue:''});
    TVBUI.quickSelector(selector, "#selection-text-area", "#loadSelectionFromTextBtn");

    selector.change(function (value) {
        GVAR_interestAreaNodeIndexes = [];
        for (var i = 0; i < value.length; i++) {
            GVAR_interestAreaNodeIndexes.push(parseInt(value[i], 10));
        }
    });

    selector.checkAll();
    modelParam.selector = selector;
}


function putModelInSelectedNodes(){
    var dynamic_id = $('#current_dynamic').val();
    var dynamic = modelParam.dynamics[dynamic_id];
    var selected = modelParam.selector.val();

    for (var i = 0; i < selected.length; i++) {
        var nodeId = selected[i];
        modelParam.modelsInNodes[nodeId] = dynamic;
    }

    modelParam.selector.setTextForSelection(dynamic.name);
    modelParam.selector.clearAll();
}

function validate(){
    var dyn_ids = [];
    var first = modelParam.modelsInNodes[0];

    for (var i = 0; i < modelParam.modelsInNodes.length; i++){
        var dyn = modelParam.modelsInNodes[i];
        if (dyn == null){
            displayMessage("node " + i + " is empty", "warningMessage");
            return false;
        }

        if (dyn.model_class !== first.model_class){
            displayMessage("all nodes must contain the same model type", "warningMessage");
            return false;
        }
        dyn_ids.push(dyn.id);
    }
    return dyn_ids;
}

function onSubmit(event){
    var dyn_ids = validate();
    if (! dyn_ids){
        event.preventDefault();
        return false;
    }
    $(this).find('input[name=dynamic_ids]').val(JSON.stringify(dyn_ids));
}


function onCanvasPick(){
    if (CONN_pickedIndex >= 0) {
        GFUNC_toggleNodeInInterestArea(CONN_pickedIndex);
        modelParam.selector.val(GVAR_interestAreaNodeIndexes);
    }
}

function onShowDynamicDetails(){
    var dynamic_id = $('#current_dynamic').val();
    doAjaxCall({
        url:'/spatial/modelparameters/regions/dynamic_detail/' + dynamic_id,
        success:function(frag){
            $('#dynamic-detail').find('.dropdown-pane').html(frag);
        }
    });
}

function main(dynamics, selectionGID){
    initSelector(selectionGID);

    var nodeIds = modelParam.selector._allValues;

    for (var i = 0; i < nodeIds.length; i++){
        var nodeId = nodeIds[i];
        modelParam.modelsInNodes[nodeId] = null;
    }
    modelParam.dynamics = dynamics;

    $('#put-model').click(putModelInSelectedNodes);
    $('#base_spatio_temporal_form').submit(onSubmit);
    $("#GLcanvas").click(onCanvasPick);
    $('#dynamic-detail').find('button').click(onShowDynamicDetails);
}

modelParam.main = main;

})();


