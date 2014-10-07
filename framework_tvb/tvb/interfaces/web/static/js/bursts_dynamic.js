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

/* globals doAjaxCall, displayMessage, MathJax, getSubmitableData */

var dynamicPage = {
    paramDefaults : {}, // information about the model parameters and their default values
    graphDefaults:{},   // information about the graph: shown state variables, state variable and axis ranges
    treeState: {},      // the state of the left input tree. Used to detect changes
    dynamic_gid: null
};

/** @module ui components module */
(function(){
    /**
     * Handles events for a group of sliders
     * @constructor
     */
    function SliderGroup(states, resetBtn, onChange){
        var self = this;
        self.onChange = onChange;
        self.states = states;
        for (var i = 0; i < states.length; i++){
            self._initSlider(states[i]);
        }
        $(resetBtn).click(function(){ self.reset(); });
    }

    SliderGroup.prototype._initSlider = function(option){
        var self = this;
        var slider = $("#slider_" + option.name);
        var input = $("#value_" + option.name);

        slider.slider({
            value: option.default,
            min: option.min,
            max: option.max,
            step: option.step,

            slide: function(ev, target) {
                input.val(target.value);
            },

            change: function(ev, target){
                input.val(target.value);
                self.onChange(option.name, target.value);
            }
        });

        input.change(function(){
            var val = parseFloat(input.val());
            if (isNaN(val) || val < option.min || val > option.max){
                val = option.default;
            }
            slider.slider('value', val);
        }).click(function(){
            input.select();
        });
    };

    SliderGroup.prototype.reset = function(){
        for (var i = 0; i < this.states.length; i++) {
            var option = this.states[i];
            var slider = $("#slider_" + option.name);
            slider.slider('value', option.default);
        }
    };

    SliderGroup.prototype.getValues = function(){
        var name2val = {};
        for (var i = 0; i < this.states.length; i++) {
            var option = this.states[i];
            var slider = $("#slider_" + option.name);
            name2val[option.name] = slider.slider('value');
        }
        return name2val;
    };

    SliderGroup.prototype.hide = function(sv_to_disable){
        for (var i = 0; i < this.states.length; i++) {
            var option = this.states[i];
            var slider = $("#slider_" + option.name);
            var input = $("#value_" + option.name);
            var enabled = sv_to_disable.indexOf(option.name) === -1;

            slider.slider(enabled? 'enable':'disable');
            input.toggle(enabled);
        }
    };

    /**
     * Dom selectors are hard coded so only one instance makes sense.
     * @constructor
     */
    function AxisGroup(state, onChange){
        var self = this;
        self.onChange = onChange;
        self.state = state;
        self.$mode = $('#mode');
        self.$svx = $('#svx');
        self.$svy = $('#svy');
        self.$slider_x = $('#slider_x_axis');
        self.$slider_y = $('#slider_y_axis');
        self.$mode.change(onChange);

        self.$svx.add(self.$svy).change(function(){
            self._initSliders();
            onChange();
        });
        $('#reset_axes').click(function() { self.reset();});
        this._initSliders();
    }

    AxisGroup.prototype._initSliders = function(){
        var x_axis_sv = this._getStateVarByName(this.$svx.val());
        var y_axis_sv = this._getStateVarByName(this.$svy.val());

        this._initAxisSlider(this.$slider_x, $('#x_range_span'), x_axis_sv);
        this._initAxisSlider(this.$slider_y, $('#y_range_span'), y_axis_sv);
    };

    AxisGroup.prototype._getStateVarByName = function(name){
        return $.grep(this.state.state_variables, function(n){
            return n.name === name;
        })[0];
    };

    AxisGroup.prototype._initAxisSlider = function(sel, span, opt){
        var self = this;
        function update_span(r) {
            span.text(r[0] + ' .. ' + r[1]);
        }
        sel.slider({
            range:true,
            min: opt.min,
            max: opt.max,
            values:[opt.lo, opt.hi],
            step: (opt.max - opt.min)/1000,
            slide: function(ev, target) {
                update_span(target.values);
            },
            change : function(ev, target){
                update_span(target.values);
                self.onChange();
            }
        });
        update_span([opt.lo, opt.hi]);
    };

    AxisGroup.prototype.reset = function(){
        var x_axis_sv = this._getStateVarByName(this.state.default_sv[0]);
        var y_axis_sv = this._getStateVarByName(this.state.default_sv[1]);

        var opt = x_axis_sv;
        this.$slider_x.slider({ min: opt.min, max: opt.max, values: [opt.lo, opt.hi] });
        opt = y_axis_sv;
        this.$slider_y.slider({ min: opt.min, max: opt.max, values: [opt.lo, opt.hi] });
        //reset mode and state var selection as well
        this.$mode.val(this.state.default_mode).change(); // change events do not fire when select's are changed by val()
        this.$svx.val(this.state.default_sv[0]).change();
        this.$svy.val(this.state.default_sv[1]).change();
    };

    AxisGroup.prototype.getValue = function(){
        return {
            mode: this.$mode.val(),
            svx: this.$svx.val(),
            svy: this.$svy.val(),
            x_range: this.$slider_x.slider('values'),
            y_range: this.$slider_y.slider('values')
        };
    };

    dynamicPage.SliderGroup = SliderGroup;
    dynamicPage.AxisGroup = AxisGroup;
})();


/** @module main */
(function(){

var DEBOUNCE_DELAY = 25;

function _url(func, tail){
    var url = '/burst/dynamic/' + func + '/' + dynamicPage.dynamic_gid;
    if (tail != null){
        url+= '/' + tail;
    }
    return url;
}

function _fetchSlidersFromServer(){
    var sliderContainer = $('#div_spatial_model_params');
    sliderContainer.empty();

    doAjaxCall({
        url: _url('sliders_fragment'),
        success: function(fragment) {
            sliderContainer.html(fragment);
            MathJax.Hub.Queue(["Typeset", MathJax.Hub, 'div_spatial_model_params']);
            setupMenuEvents(sliderContainer);

            dynamicPage.paramSliders = new dynamicPage.SliderGroup(dynamicPage.paramDefaults, '#reset_sliders', onParameterChanged);
            dynamicPage.stateVarsSliders = new dynamicPage.SliderGroup(dynamicPage.graphDefaults.state_variables, '#reset_state_variables', onGraphChanged);
            dynamicPage.axisControls = new dynamicPage.AxisGroup(dynamicPage.graphDefaults, onGraphChanged);
            _onParameterChanged();
            _disable_active_sv_slider();
        }
    });
}

function onModelChanged(name){
    doAjaxCall({
        url: _url('model_changed', name) ,
        success: function(data){
            data = JSON.parse(data);
            dynamicPage.paramDefaults = data.params;
            dynamicPage.graphDefaults = data.graph_params;
            _fetchSlidersFromServer();
        }
    });
}

function _redrawPhasePlane(data){
    data = JSON.parse(data);
    dynamicPage.phasePlane.draw(data);
    dynamicPage.phasePlane.clearTrajectories();
    var axisState = dynamicPage.axisControls.getValue();
    dynamicPage.phasePlane.setLabels(axisState.svx, axisState.svy);
    dynamicPage.phasePlane.setPlotLabels($.map(dynamicPage.graphDefaults.state_variables, function(d){return d.name;}) );
}

function _onParameterChanged(){
    doAjaxCall({
        url: _url('parameters_changed'),
        data: {params: JSON.stringify(dynamicPage.paramSliders.getValues())},
        success : _redrawPhasePlane
    });
}

// Resetting a slider group will trigger change events for each slider. The handler does a slow ajax so debounce the handler
var onParameterChanged = $.debounce(DEBOUNCE_DELAY, _onParameterChanged);

function _disable_active_sv_slider(){
    var axis_state = dynamicPage.axisControls.getValue();
    dynamicPage.stateVarsSliders.hide([axis_state.svx, axis_state.svy]);
}

function _onGraphChanged(){
    var graph_state = dynamicPage.axisControls.getValue();
    graph_state.state_vars = dynamicPage.stateVarsSliders.getValues();
    _disable_active_sv_slider();
    doAjaxCall({
        url: _url('graph_changed'),
        data: { graph_state: JSON.stringify(graph_state)},
        success : _redrawPhasePlane
    });
}

// see onParameterChanged
var onGraphChanged = $.debounce(DEBOUNCE_DELAY, _onGraphChanged);

function onTrajectory(x, y){
    doAjaxCall({
        url: _url('trajectory'),
        data: {x:x, y:y},
        success:function(data){
            data = JSON.parse(data);
            if (data.finite) {
                dynamicPage.phasePlane.drawTrajectory(data.trajectory);
                dynamicPage.phasePlane.drawSignal(data.signals);
            }else{
                displayMessage('Trajectory contains infinities. Try to decrease the integration step.', 'warningMessage');
            }
        }
    });
}


// Throttle the rate of trajectory creation. In a burst of mouse clicks some will be ignored.
// Ignore trailing events. Without this throttling the server overwhelms and numexpr occasionally segfaults.
var onTrajectory = $.throttle(500, true, onTrajectory);

function onSubmit(event){
    var name = $('#dynamic_name').val().trim();
    if (name.length ) {
        doAjaxCall({
            url: _url('submit', name),
            success: function(){
                displayMessage('Dynamic saved', 'importantMessage');
            }
        });
    }else{
        displayMessage('A name is required', 'errorMessage');
    }
    event.preventDefault();
}

function onIntegratorChanged(state){
    doAjaxCall({
        url: _url('integrator_changed'),
        data: state,
        success:function(){
            dynamicPage.phasePlane.clearTrajectories();
        }
    });
}

// Event debouncing makes less sense now that the requests have been made blocking.
var debouncedOnIntegratorChanged = $.debounce( DEBOUNCE_DELAY, onIntegratorChanged);

// Detect changes by doing a tree diff. This diff is simple but relies on a specific tree structure.
function onTreeChange(){
    var state = getSubmitableData('left_input_tree');
    var previous = dynamicPage.treeState;
    if (state.model_type !== previous.model_type){
        onModelChanged(state.model_type);
    }else if (state.dynamic_name === previous.dynamic_name){
        // Name has not changed. The change is in the integrator subtree
        debouncedOnIntegratorChanged(state);
    }
    dynamicPage.treeState = state;
}

function main(dynamic_gid){
    dynamicPage.dynamic_gid = dynamic_gid;
    $('.field-adapters').hide(); // hide expand range buttons. Careful as this class is used for other things as well
    // listen for changes of the left tree
    $('#left_input_tree').find('input').add('select').change(onTreeChange);
    $('#base_spatio_temporal_form').submit(onSubmit);
    onTreeChange();
    dynamicPage.phasePlane = new TVBUI.PhasePlane(onTrajectory);
}

dynamicPage.main = main;

})();
