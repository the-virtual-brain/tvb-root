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

// This file uses some js patterns.
// Prototypes. See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Inheritance_and_the_prototype_chain
// The 'module pattern' : Immediately invoked anonymous functions
// "use strict" strict js semantics
// Jquery custom events pub/sub

/**
 * Module of tvb ui components
 * @module {TVBUI}
 */
var TVBUI = TVBUI || {};

/**
 * depends on jquery and displayMessage
 * @module
 */
(function($, displayMessage, TVBUI){
"use strict";
/**
 * @constructor
 * @param dom selector or dom node
 * @param settings selectors for sub-components. see defaults
 */
function RegionSelectComponent(dom, settings){
    var $dom = $(dom);
    var self = this;
    settings = $.extend({}, self.defaults, settings);
    self.$dom = $dom;
    self.settings = settings;
    self._selectedValues = [];
    self._selectedIndices = [];
    self._namedSelections = [];
    // array with the values of all check boxes
    self._allValues = [];
    self._labels = [];
    // save dom variables, set up listeners
    self._boxes = $dom.find(settings.boxesSelector);
    self._textBox = $dom.find(settings.textboxSelector);
    self._dropDown = $dom.find(settings.selectionDropdown);
    self._dropDownOptions = self._dropDown.find('option');

    self._boxes.change(function(){self._onchange(this);});
    self._dropDown.change(function(){
        self._set_val(JSON.parse(this.value));
        self.$dom.trigger("selectionChange", [self._selectedValues.slice()]);
    });
    $dom.find(settings.checkSelector).click(function(){ self.checkAll();} );
    $dom.find(settings.uncheckSelector).click(function(){ self.clearAll();} );
    $dom.find(settings.saveSelectionButtonSelector).click(function(){ self._onNewSelection(); });

    $dom.find(settings.applySelector).click(function(){
        self.$dom.trigger("selectionApplied", [self._selectedValues.slice()]);
    });
    this.dom2model();
}

RegionSelectComponent.prototype.defaults = {
    applySelector: '.action-view',
    checkSelector: '.action-all-on',
    uncheckSelector: '.action-all-off',
    boxesSelector: 'input[type=checkbox]',
    textboxSelector : 'input[type=text]',
    saveSelectionButtonSelector : '.action-store',
    selectionDropdown: '.selection-toolbar > select'
};

/**
 * Subscribe a function to the selection change event 
 */
RegionSelectComponent.prototype.change = function(fn){
    this.$dom.on("selectionChange", function(_event, arg){ fn(arg); });
};

/**
 * Unbind all event handlers
 */
RegionSelectComponent.prototype.destroy = function(){
    this._boxes.off();
    this._dropDown.off();
    this.$dom.find(this.settings.checkSelector).off();
    this.$dom.find(this.settings.uncheckSelector).off();
    this.$dom.find(this.settings.saveSelectionButtonSelector).off();
    this.$dom.find(this.settings.applySelector).off();
    this.$dom.off("selectionChange");
};

/**
 * Updates the model from dom values
 * @private
 */
RegionSelectComponent.prototype.dom2model = function(){
    var self = this;
    self._allValues = [];
    self._selectedValues = [];
    self._selectedIndices = [];
    self._namedSelections = [];

    this._boxes.each(function(idx, el){
        self._allValues.push(el.value);
        // assumes the parent element is the label
        self._labels.push($(this).parent().text().trim());
        if(el.checked){
            self._selectedValues.push(el.value);
            self._selectedIndices.push(idx);
        }
    });
    this._dropDownOptions.each(function(i, el){
        if(i !== 0){
            var $el = $(el);
            self._namedSelections.push([$el.text(), $el.val()]);
        }
    });
};

RegionSelectComponent.prototype._updateDecoration = function(el){
    // here we assume dom structure
    $(el).parent().toggleClass("selected", el.checked);
};

/**
 * Updates the dom from the selection model
 * @private
 */
RegionSelectComponent.prototype.selectedValues2dom = function(){
    var self = this;
    this._boxes.each(function(_, el){
        var idx = self._selectedValues.indexOf(el.value);
        el.checked = idx !== -1;
        self._updateDecoration(el);
    });
    self._dropDownOptions = self._dropDown.find('option');
};

/**
 * Handler for save selection dom event
 * @private
 */
RegionSelectComponent.prototype._onNewSelection = function(){
    var self = this;
    var name = $.trim(self._textBox.val());
    if (name !== ""){
        // add to model
        self._namedSelections.push([name, self._selectedValues.slice()]);
        self._textBox.val('');
        // do not update the selection box. let a event listener decide
        self.$dom.trigger("newSelection", [name, self._selectedValues.slice()]);
    }else{
        displayMessage("Selection name must not be empty.", "errorMessage");
    }
};

/**
 * Handler for checkbox change dom event
 * @private
 */
RegionSelectComponent.prototype._onchange = function(el){
    this.dom2model();
    this._dropDown.val("[]");
    this._updateDecoration(el);
    this.$dom.trigger("selectionChange", [this._selectedValues.slice()]);
};

/**
 * Sets the selection without triggering events
 * @private
 */
RegionSelectComponent.prototype._set_val = function(arg){
    this._selectedValues = [];
    this._selectedIndices =[];
    for(var i=0; i < arg.length; i++){
        // convert vals to string (as in dom)
        var val = arg[i].toString();
        // filter bad values
        var idx = this._allValues.indexOf(val);
        if( idx !== -1){
            this._selectedValues.push(val);
            this._selectedIndices.push(idx);
        }else{
            console.warn("bad selection" + val);
        }
    }
    this.selectedValues2dom();
};

/**
 * Sets the selection without triggering events
 * @private
 */
RegionSelectComponent.prototype._set_indices = function(arg){
    this._selectedValues = [];
    this._selectedIndices =[];
    for(var i=0; i < arg.length; i++){
        var idx = arg[i];
        // filter bad values
        if( idx >= 0 && idx < this._allValues.length){
            this._selectedValues.push(this._allValues[idx]);
            this._selectedIndices.push(idx);
        }else{
            console.warn("bad index selection" + arg[i]);
        }
    }
    this.selectedValues2dom();
};

/**
 * Gets the selected values if no argument is given
 * Sets the selected values if an array of values is given as argument
 */
RegionSelectComponent.prototype.val = function(arg){
    if(arg == null){
        return this._selectedValues.slice();
    }else{
        this._set_val(arg);
        this._dropDown.val("[]");
        this.$dom.trigger("selectionChange", [this._selectedValues.slice()]);
    }
};

/**
 * Return the selected indices
 * @returns {Array}
 */
RegionSelectComponent.prototype.selectedIndices = function(arg){
    if (arg==null) {
        return this._selectedIndices.slice();
    }else{
        this._set_indices(arg);
        this._dropDown.val("[]");
        this.$dom.trigger("selectionChange", [this._selectedValues.slice()]);

    }
};

RegionSelectComponent.prototype.selectedLabels = function(){
    var ret = [];
    for(var i = 0 ; i < this._selectedIndices.length; i++){
        var selected_idx = this._selectedIndices[i];
        ret.push(this._labels[selected_idx]);
    }
    return ret;
};

RegionSelectComponent.prototype.clearAll = function(){
    this.val([]);
};

RegionSelectComponent.prototype.checkAll = function(){
    this.val(this._allValues);
};

/**
 * This is a component associated with a selection component.
 * It allows selection using a text representation of the labels
 * @param regionSelectComponent
 * @param textDom
 * @param buttonDom
 * @param onerror
 * @constructor
 */
function QuickSelectComponent(regionSelectComponent, textDom, buttonDom, onerror){
    var self = this;
    self._text = $(textDom);
    self._onerror = onerror;
    self._regionSelectComponent = regionSelectComponent;
    self._selected_to_text();

    regionSelectComponent.change(function(values){
        self._selected_to_text();
    });

    $(buttonDom).click(function(){
        var values = self._parse(self._text.val());
        if(values != null){
            self._regionSelectComponent.val(values);
        }
    });
}

QuickSelectComponent.prototype._selected_to_text = function(){
    this._text.val( '[' + this._regionSelectComponent.selectedLabels().join(', ') + ']' );
};

QuickSelectComponent.prototype._parse = function(text){
    // we could be more lenient and split on spaces as well, and ignore case
    var nodes = text.replace('[', '').replace(']', '').split(',');
    var values = [], badLabels = [];

    for (var i = 0; i < nodes.length; i++){
        var node = nodes[i].trim();
        // labels have the same order as _allvalues
        var label_idx = this._regionSelectComponent._labels.indexOf(node);
        if (label_idx !== -1){
            values.push(this._regionSelectComponent._allValues[label_idx]);
        }else{
            badLabels.push(node);
        }
    }
    if (badLabels.length > 0 ){
        this._onerror('Invalid node names:' + badLabels.join(','));
        return null;
    }else{
        return values;
    }
};

/**
 * This inherits from RegionSelectComponent and adds arbitrary text for each channel
 * It is a view. It holds no model. The text values are not remembered.
 * Debatable if inheritance is worth it. If it is not merge this behaviour in parent
 * The flavor of inheritance used is described below
 * @constructor
 * @extends RegionSelectComponent
 */
function TextGridSelectComponent(dom, settings){
    // calling a ctor without new creates no new object but attaches everything to this
    // so calling it with the current (empty) object initializes as the super ctor would have
    RegionSelectComponent.call(this, dom, settings);
    if (this.settings.emptyValue === undefined) { this.settings.emptyValue = 0; }
    // inject spans
    this._boxes.each(function(){
        $("<span></span>").addClass("node-scale").text(settings.emptyValue).appendTo($(this).parent());
    });
    this._spans = this.$dom.find("span.node-scale");
}

// proto chain setup TextGridSelectComponent.prototype = {new empty obj} -> RegionSelectComponent.prototype
// Object.create is needed TextGridSelectComponent.prototype = RegionSelectComponent.prototype;
// would have had the effect that changing TextGridSelectComponent.prototype would've changed RegionSelectComponent.prototype
TextGridSelectComponent.prototype = Object.create(RegionSelectComponent.prototype);

TextGridSelectComponent.prototype.setTextForSelection = function(nr){
    for(var i = 0; i < this._selectedIndices.length; i++){
        var selected_idx = this._selectedIndices[i];
        var sp = $(this._spans[selected_idx]);
        sp.text(nr);
        sp.toggleClass("node-scale-selected", nr !== 0);
    }
};

/**
 * @param nrs a dict of {node_id: label} Or an array where the index is implicitly the node id
 */
TextGridSelectComponent.prototype.setGridText = function(nrs){
    var vals = Object.keys(nrs);
    for (var i = 0; i < vals.length; i++){
        var id = vals[i];
        var idx = this._allValues.indexOf(id); // convert node id to node index
        if( idx !== -1){
            var sp = $(this._spans[idx]);
            sp.text('' + nrs[id]);
            sp.toggleClass("node-scale-selected", nrs[id] !== this.settings.emptyValue);
        } else {
            console.warn("bad selection" + id);
        }
    }
};

/**
 * This is a component managing a mode and a state variable select
 * @constructor
 */
function ModeAndStateSelectComponent(dom, id){
    var self = this;
    var $dom = $(dom);
    self._id = id;
    self._modeSelect = $dom.find('.mode-select');
    self._stateVarSelect = $dom.find('.state-variable-select');
}

/**
 * Subscribe a function to the mode change event
 */
ModeAndStateSelectComponent.prototype.modeChanged = function(fn){
    var self = this;
    self._modeSelect.change(function(){
        fn(self._id, $(this).val());
    });
};

/**
 * Subscribe a function to the state variable change event
 */
ModeAndStateSelectComponent.prototype.stateVariableChanged = function(fn){
    var self = this;
    self._stateVarSelect.change(function(){
        fn(self._id, $(this).val());
    });
};

// @exports
TVBUI.RegionSelectComponent = RegionSelectComponent;
TVBUI.TextGridSelectComponent = TextGridSelectComponent;
TVBUI.QuickSelectComponent = QuickSelectComponent;
TVBUI.ModeAndStateSelectComponent = ModeAndStateSelectComponent;

})($, displayMessage, TVBUI);  //depends

/**
 * This module has factory methods for the selection components.
 * It also synchronizes named selection with the server for the created components
 * @module
 */
(function($, displayMessage, doAjaxCall, TVBUI){
"use strict";
/**
 * Makes a selection component save selections on the server
 * @private
 */
function createServerSynchronizingSelector(component, filterGid){
    function getSelections() {
        doAjaxCall({type: "POST",
            async: false,   
            url: '/flow/get_available_selections',
            data: {'datatype_gid': filterGid},
            success: function(r) {
                component._dropDown.empty().append(r);
                component._dropDown.val('[' + component._selectedValues.join(', ')     + ']');
            } ,
            error: function(r) {
                displayMessage("Error while retrieving available selections.", "errorMessage");
            }
        });
    }

    getSelections();

    component.$dom.on("newSelection", function(_ev, name, selection){
        doAjaxCall({
            type: "POST",
            url: '/flow/store_measure_points_selection/' + name,
            data: {'selection': JSON.stringify(selection),
                   'datatype_gid': filterGid},
            success: function(r) {
                var response = $.parseJSON(r);
                if (response[0]) {
                    getSelections();
                    displayMessage(response[1], "infoMessage");
                } else {
                    displayMessage(response[1], "errorMessage");
                }
            },
            error: function() {
                displayMessage("Selection was not saved properly.", "errorMessage");
            }
        });
    });
}

/**
 * Creates a selection component.
 * Synchronizes named selections with the server.
 * @param dom selector for the container div
 * @param [settings]
 * @returns {TVBUI.RegionSelectComponent}
 */
TVBUI.regionSelector = function(dom, settings){
    var component =  new TVBUI.RegionSelectComponent(dom, settings);
    createServerSynchronizingSelector(component, settings.filterGid);
    return component;
};

/**
 * Creates a selection component that associates a number with each label
 * Synchronizes named selections with the server.
 * @param dom selector for the container div
 * @param [settings]
 * @returns {TVBUI.TextGridSelectComponent}
 */
TVBUI.textGridRegionSelector = function(dom, settings){
    var component =  new TVBUI.TextGridSelectComponent(dom, settings);
    createServerSynchronizingSelector(component, settings.filterGid);
    return component;
};

/**
 * Creates a quick selection component wrapping a selection component
 * Synchronizes named selections with the server.
 * @param regionSelectComponent A {TVBUI.RegionSelectComponent}
 * @param textDom a selector of the text area containing the text represenatation of the selection
 * @param buttonDom a selector for the apply selection button
 * @returns {TVBUI.QuickSelectComponent}
 */
TVBUI.quickSelector = function(regionSelectComponent, textDom, buttonDom){
    return new TVBUI.QuickSelectComponent(regionSelectComponent,
        textDom, buttonDom, function(txt) {displayMessage(txt, "errorMessage");});
};

/**
 * Creates a component that manages mode and state variable selects
 * @param dom selector of container element
 * @param id  some identification of the associated time series
 * @param onModeChange callback
 * @param onStateVariableChange callback
 * @returns {TVBUI.ModeAndStateSelectComponent}
 */
TVBUI.modeAndStateSelector = function(dom, id){
    return new TVBUI.ModeAndStateSelectComponent(dom, id);
};

})($, displayMessage, doAjaxCall, TVBUI); // depends


