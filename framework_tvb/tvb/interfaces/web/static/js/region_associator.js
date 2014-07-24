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

/* globals TVBUI, GVAR_interestAreaNodeIndexes, CONN_pickedIndex, GFUNC_toggleNodeInInterestArea */

(function(TVBUI, GFUNC_toggleNodeInInterestArea){

/**
 * A view for associating some values to connectivity nodes.
 * @param settings An object with these keys:
 *                 selectionGID: the gid of the connectivity,
 *                 onPut: a callback receiving a list of selected node ids. It should return a text to place in the ui,
 *                 prepareSubmitData: a callback that should return the data to submit or null to cancel the submit
 * @constructor
 */
function RegionAssociatorView(settings){
    var self = this;
    self.settings = settings;
    self.selector = self.createSelector(settings.selectionGID);

    $('#put-model').click(function(){self._onPut();});
    $('#base_spatio_temporal_form').submit(function(e){self._onSubmit(e);});
    $("#GLcanvas").click(function(){self._onCanvasPick();});
}

/**
 * Create a text grid region selector that is synchronized with the GVAR_interestAreaNodeIndexes global
 * @returns {TVBUI.TextGridSelectComponent}
 */
RegionAssociatorView.prototype.createSelector = function(selectionGID){
    var selector = TVBUI.textGridRegionSelector("#channelSelector", {filterGid: selectionGID, emptyValue:''});
    TVBUI.quickSelector(selector, "#selection-text-area", "#loadSelectionFromTextBtn");

    selector.change(function (value) {
        GVAR_interestAreaNodeIndexes = [];
        for (var i = 0; i < value.length; i++) {
            GVAR_interestAreaNodeIndexes.push(parseInt(value[i], 10));
        }
    });
    selector.checkAll();
    return selector;
};

RegionAssociatorView.prototype._onCanvasPick = function(){
    if (CONN_pickedIndex >= 0) {
        GFUNC_toggleNodeInInterestArea(CONN_pickedIndex);
        this.selector.val(GVAR_interestAreaNodeIndexes);
    }
};

RegionAssociatorView.prototype._onSubmit = function(event){
    var data = this.settings.prepareSubmitData();
    if (data != null) {
        $(event.target).find('input[name=node_values]').val(JSON.stringify(data));
    }else{
        event.preventDefault();
        return false;
    }
};

RegionAssociatorView.prototype._onPut = function(){
    var text = this.settings.onPut(this.selector.val());
    this.selector.setTextForSelection(text);
    this.selector.clearAll();
};

RegionAssociatorView.prototype.setGridText = function(text){
    this.selector.setGridText(text);
};

TVBUI.RegionAssociatorView = RegionAssociatorView;

})(TVBUI, GFUNC_toggleNodeInInterestArea);