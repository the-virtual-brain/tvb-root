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

/* globals TVBUI, GVAR_interestAreaNodeIndexes, CONN_pickedIndex, GFUNC_toggleNodeInInterestArea */

(function(TVBUI, GFUNC_toggleNodeInInterestArea){

/**
 * A view for associating some values to connectivity nodes.
 * The synchronization with the webgl view happens via the global GVAR_interestAreaNodeIndexes
 * GVAR_interestAreaNodeIndexes has the *indices* of the selected nodes; indices in a hemispheric ordered view of the connectivity.
 * The mapping idx->id is done assumming that the check boxes of the selection component appear in the dom in index order.
 * So the first checkbox has idx 0 and id the value attr
 * @param settings An object with these keys:
 *                 selectionGID: the gid of the connectivity,
 *                 onPut: a callback receiving a list of selected node ids. It should return a text to place in the ui,
 *                 prepareSubmitData: a callback that should return the data to submit or null to cancel the submit
 * @constructor
 */
function RegionAssociatorView(settings){
    let self = this;
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
    let selector = TVBUI.textGridRegionSelector("#channelSelector", {filterGid: selectionGID, emptyValue: ''});
    TVBUI.quickSelector(selector, "#selection-text-area", "#loadSelectionFromTextBtn");

    selector.change(function () {
        GVAR_interestAreaNodeIndexes = selector.selectedIndices();
        GFUNC_updateLeftSideVisualization();
    });
    selector.checkAll();
    return selector;
};

RegionAssociatorView.prototype._onCanvasPick = function(){
    if (CONN_pickedIndex >= 0) {
        GFUNC_toggleNodeInInterestArea(CONN_pickedIndex);
        this.selector.selectedIndices(GVAR_interestAreaNodeIndexes);
    }
};

RegionAssociatorView.prototype._onSubmit = function(event){
    const data = this.settings.prepareSubmitData();
    if (data !== null) {
        $(event.target).find('input[name=node_values]').val(JSON.stringify(data));
    }else{
        event.preventDefault();
        return false;
    }
};

RegionAssociatorView.prototype._onPut = function(){
    const text = this.settings.onPut(this.selector.val());
    this.selector.setTextForSelection(text);
    this.selector.clearAll();
};

RegionAssociatorView.prototype.setGridText = function(text){
    this.selector.setGridText(text);
};

TVBUI.RegionAssociatorView = RegionAssociatorView;

})(TVBUI, GFUNC_toggleNodeInInterestArea);