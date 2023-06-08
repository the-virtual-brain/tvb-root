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

/* globals doAjaxCall, displayMessage, MathJax, getSubmitableData */

var dynamicPage = {
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
     * @constructor
     */
    function AxisControls(state, svDrop, slider, span, onChange){
        var self = this;
        self.onChange = onChange;
        self.state = state;
        self.$sv = $(svDrop);
        self.$slider = $(slider);
        self.$span = $(span);

        self.$sv.change(function(){
            self.val(self.$sv.val());
            onChange();
        });

        self.$slider.slider({
            range:true,
            slide: function(ev, target) {
                self._updateSpan(target.values);
            },
            change : function(ev, target){
                self._updateSpan(target.values);
                self.onChange();
            }
        });
    }

    AxisControls.prototype._getStateVarByName = function(name){
        return $.grep(this.state.state_variables, function(n){
            return n.name === name;
        })[0];
    };

    AxisControls.prototype._updateSpan = function(r){
        this.$span.text(r[0] + ' .. ' + r[1]);
    };

    AxisControls.prototype.val = function(sv){
        if (sv == null) {
            return {sv: this.$sv.val(), range: this.$slider.slider('values')};
        }else{
            var opt = this._getStateVarByName(sv);
            this.$sv.val(sv);
            this.$slider.slider({ min: opt.min, max: opt.max, values: [opt.lo, opt.hi], step: (opt.max - opt.min)/1000 });
            this._updateSpan([opt.lo, opt.hi]);
        }
    };


    /**
     * Dom selectors are hard coded so only one instance makes sense.
     * deprecated: move logic to plane controller
     * @constructor
     */
    function AxisGroup(state, onChange){
        var self = this;
        self.onChange = onChange;
        self.state = state;
        self.$mode = $('#mode');
        self.ax = new AxisControls(state, '#svx', '#slider_x_axis', '#x_range_span', onChange);
        self.ay = new AxisControls(state, '#svy', '#slider_y_axis', '#y_range_span', onChange);
        self.ax.val(this.state.default_sv[0]);
        self.ay.val(this.state.default_sv[1]);
        self.$mode.change(onChange);
        $('#reset_axes').click(function() { self.reset();});
    }

    AxisGroup.prototype.reset = function(){
        this.$mode.val(this.state.default_mode);
        this.ax.val(this.state.default_sv[0]);
        this.ay.val(this.state.default_sv[1]);
    };

    AxisGroup.prototype.getValue = function(){
        var axv = this.ax.val();
        var ayv = this.ay.val();
        return {
            mode: this.$mode.val(),
            svx: axv.sv, svy: ayv.sv,
            x_range: axv.range, y_range: ayv.range
        };
    };

    dynamicPage.SliderGroup = SliderGroup;
    dynamicPage.AxisGroup = AxisGroup;
    dynamicPage.AxisControls = AxisControls;
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

/**
 * Server connected phase view.
 * Handles graph state and trajectories.
 * @constructor
 */
function PhasePlaneController(graph_defaults, phasePlane) {
    var self = this;
    this.graph_defaults = graph_defaults;  // information about the graph: shown state variables, state variable and axis ranges
    this.traj_starts = [];                 // trajectory starting points. Kept to resubmit trajectory computation if model params change
    // the trajectories/signals raw data.
    // It is idiomatic in d3 not to draw incrementally but to update the data set.
    // Thus we keep the dataset because we have to update if a new traj is added.
    this.trajectories = [];
    this.phasePlane = phasePlane;
    // see onParameterChanged
    var onGraphChanged = $.debounce(DEBOUNCE_DELAY, function(){self._onGraphChanged();});
    this.stateVarsSliders = new dynamicPage.SliderGroup(graph_defaults.state_variables, '#reset_state_variables', onGraphChanged);
    this.axisControls = new dynamicPage.AxisGroup(graph_defaults, onGraphChanged);
    $('#reset_trajectories').click(function(){self._deleteTrajectories();});
    // Throttle the rate of trajectory creation. In a burst of mouse clicks some will be ignored.
    // Ignore trailing events. Without this throttling the server overwhelms and numexpr occasionally segfaults.
    var onTrajectory = $.throttle(500, true, function(x, y){self.onTrajectory(x, y);});
    this.phasePlane.onClick = onTrajectory;
    $(this.phasePlane.svg[0]).show();
    //clear all trajectories
    this._deleteTrajectories();
    this._disable_active_sv_slider();

    this.intStepsSlider = $('#slider_integration_steps');
    var intStepsSpan = $('#span_integration_steps').text(graph_defaults.integration_steps.default);
    this.intStepsSlider.slider({
        value: graph_defaults.integration_steps.default,
        min: graph_defaults.integration_steps.min,
        max: graph_defaults.integration_steps.max,
        step: 5,
        slide : function(ev, target){intStepsSpan.text(target.value);},
        change : function(ev, target){self._redrawTrajectories();}
    });
}

PhasePlaneController.prototype.draw = function(data){
    this._redrawPhasePlane(data);
    this._redrawTrajectories();
};

PhasePlaneController.prototype.onIntegratorChanged = function(){
    this._redrawTrajectories();
};

PhasePlaneController.prototype._redrawPhasePlane = function(data){
    data = JSON.parse(data);
    this.phasePlane.draw(data);
    var axis_state = this.axisControls.getValue();
    this.phasePlane.setLabels(axis_state.svx, axis_state.svy);
    this.phasePlane.setPlotLabels($.map(this.graph_defaults.state_variables, function(d){return d.name;}) );
};

PhasePlaneController.prototype._disable_active_sv_slider = function(){
    var axis_state = this.axisControls.getValue();
    this.stateVarsSliders.hide([axis_state.svx, axis_state.svy]);
};

PhasePlaneController.prototype._onGraphChanged = function(){
    var self = this;
    var axis_state = this.axisControls.getValue();
    axis_state.state_vars = this.stateVarsSliders.getValues();
    this._disable_active_sv_slider();
    doAjaxCall({
        url: _url('graph_changed'),
        data: { graph_state: JSON.stringify(axis_state)},
        success : function(data){
            self._redrawPhasePlane(data);
            self._redrawTrajectories();
        }
    });
};

PhasePlaneController.prototype._deleteTrajectories = function(){
    this.trajectories = [];
    this.traj_starts = [];
    this.phasePlane.drawTrajectories([]);
    this.phasePlane.drawSignal([]);
};

function _trajectories_rpc(starting_points, integration_steps, success){
    doAjaxCall({
        url: _url('trajectories'),
        data: {
            starting_points: JSON.stringify(starting_points),
            integration_steps: integration_steps
        },
        success:function(data){
            data = JSON.parse(data);
            if (data.finite) {
                success(data);
            }else{
                displayMessage('Trajectory contains infinities. Try to decrease the integration step.', 'warningMessage');
            }
        }
    });
}

PhasePlaneController.prototype.onTrajectory = function(x, y){
    var self = this;
    var start_state = this.stateVarsSliders.getValues();
    var axis_state = this.axisControls.getValue();
    start_state[axis_state.svx] = x;
    start_state[axis_state.svy] = y;
    var integration_steps = this.intStepsSlider.slider('value');
    _trajectories_rpc([start_state], integration_steps, function(data){
        self.traj_starts.push(start_state);
        self.trajectories.push(data.trajectories[0]);
        self.phasePlane.drawTrajectories(self.trajectories);
        self.phasePlane.drawSignal(data.signals);
    });
};

PhasePlaneController.prototype._redrawTrajectories = function(){
    var self = this;
    if (this.traj_starts.length === 0){
        return;
    }
    var integration_steps = this.intStepsSlider.slider('value');
    _trajectories_rpc(this.traj_starts, integration_steps, function(data){
        self.trajectories = data.trajectories;
        self.phasePlane.drawTrajectories(self.trajectories);
        self.phasePlane.drawSignal(data.signals);
    });
};

PhasePlaneController.prototype.destroy = function(){
    $(this.phasePlane.svg[0]).hide();
};

function PhaseGraphController(graph_defaults, phaseGraph) {
    var self = this;
    this.graph_defaults = graph_defaults;
    this.phaseGraph = phaseGraph;

    var onGraphChanged = $.debounce(DEBOUNCE_DELAY, function(){self._onGraphChanged();});

    self.$mode = $('#mode');
    self.ax = new dynamicPage.AxisControls(graph_defaults, '#svx', '#slider_x_axis', '#x_range_span', onGraphChanged);
    self.ax.val(graph_defaults.default_sv[0]);
    self.$mode.change(onGraphChanged);
    $('#reset_axes').click(function() {
        self.$mode.val(graph_defaults.default_mode);
        self.ax.val(graph_defaults.default_sv[0]);
    });

    $(this.phaseGraph.svg[0]).show();
}

PhaseGraphController.prototype._onGraphChanged = function(){
    var self = this;
    var axv = this.ax.val();
    var axis_state = {
        mode: this.$mode.val(),
        svx: axv.sv,
        x_range: axv.range
    };

    doAjaxCall({
        url: _url('graph_changed'),
        data: { graph_state: JSON.stringify(axis_state)},
        success : function(data){
            self.draw(data);
        }
    });
};

PhaseGraphController.prototype.draw = function(data){
    data = JSON.parse(data);
    this.phaseGraph.draw(data);
    var sv = this.ax.val().sv;
    this.phaseGraph.setLabels(sv, sv + " '");
};

PhaseGraphController.prototype.onIntegratorChanged = function(){

};

PhaseGraphController.prototype.destroy = function(){
    $(this.phaseGraph.svg[0]).hide();
};

function _initialize_grafic(paramDefaults, graphDefaults){
    if (dynamicPage.grafic != null){
        dynamicPage.grafic.destroy();
    }
    if (graphDefaults.state_variables.length > 1) {
        dynamicPage.grafic = new PhasePlaneController(graphDefaults, dynamicPage.phasePlane);
    }else{
        dynamicPage.grafic = new PhaseGraphController(graphDefaults, dynamicPage.phaseGraph);
    }

    if (paramDefaults.length) {
        dynamicPage.paramSliders = new dynamicPage.SliderGroup(paramDefaults, '#reset_sliders', onParameterChanged);
    }else{
        //model has no params; schedule a draw
        //this is a mocup have to get data from server;
        //dynamicPage.grafic.draw();
    }
}

function onModelChanged(name){
    doAjaxCall({
        url: _url('model_changed', name) ,
        success: function(data){
            data = JSON.parse(data);

            var sliderContainer = $('#div_spatial_model_params');
            renderWithMathjax(sliderContainer, data.model_param_sliders_fragment, true);
            setupMenuEvents(sliderContainer);

            var axisSliderContainer = $('#div_phase_plane_settings');
            renderWithMathjax(axisSliderContainer, data.axis_sliders_fragment, true);
            setupMenuEvents(axisSliderContainer);

            _initialize_grafic(data.params, data.graph_params);
        }
    });
}

function _onParameterChanged(){
    doAjaxCall({
        url: _url('parameters_changed'),
        data: {params: JSON.stringify(dynamicPage.paramSliders.getValues())},
        success : function(data){
            dynamicPage.grafic.draw(data);
        }
    });
}

// Resetting a slider group will trigger change events for each slider. The handler does a slow ajax so debounce the handler
var onParameterChanged = $.debounce(DEBOUNCE_DELAY, _onParameterChanged);

function onSubmit(event){
    var name = $('#dynamic_name').val().trim();
    if (name.length ) {
        doAjaxCall({
            url: _url('submit', name),
            success: function(res){
                res = JSON.parse(res);
                if(res.saved){
                    displayMessage('Dynamic saved', 'importantMessage');
                }else{
                    displayMessage(res.msg, 'errorMessage');
                }
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
        success: function(){ dynamicPage.grafic.onIntegratorChanged();}
    });
}

// Event debouncing makes less sense now that the requests have been made blocking.
var debouncedOnIntegratorChanged = $.debounce( DEBOUNCE_DELAY, onIntegratorChanged);

function onLeftInputTreeChange(){
    var state = getSubmitableData('left_input_tree');
    onModelChanged(state.model);
}

function onIntegratorInputTreeChange(){
    var state = getSubmitableData('integrator_input_tree');
    debouncedOnIntegratorChanged(state);
}

function main(dynamic_gid){
    dynamicPage.dynamic_gid = dynamic_gid;
    $('.field-adapters').hide(); // hide expand range buttons. Careful as this class is used for other things as well
    // listen for changes of the input trees
    $('#left_input_tree').find('select').change(onLeftInputTreeChange); // intentionally omit <input>. We only need to listen for model type changes
    // $('#integrator_input_tree').find('input, select').change(onIntegratorInputTreeChange);
    $('#base_spatio_temporal_form').submit(onSubmit);
    onLeftInputTreeChange();
    dynamicPage.phasePlane = new TVBUI.PhasePlane('#phasePlane');
    dynamicPage.phaseGraph = new TVBUI.PhaseGraph('#phaseGraph');
}

dynamicPage.main = main;

})();

function setIntegratorParamAndRedrawChart(methodToCall, fieldName, fieldValue, type) {
    let currentParam = fieldName + '=' + fieldValue;
    let url = deploy_context + methodToCall + '/' + dynamicPage.dynamic_gid + '/' + type + '?' + currentParam;
    $.ajax({
        url: url,
        type: 'POST',
        success: function () {
            plotEquation();
        }
    })
}

function plotEquation(subformDiv = null) {
    dynamicPage.grafic._redrawTrajectories();
}

function setEventsOnFormFields(param, div_id) {
    $('#' + div_id + ' input:not([type=radio])').each(function () {
        let events = $._data(document.getElementById($(this).attr('id')),'events')
        if (!events || !("change" in events)){
            $(this).change(function () {
                setIntegratorParamAndRedrawChart('integrator_parameters_changed', this.name, this.value, param)
            })
        }
    });
}

function prepareRefreshSubformUrl(currentElem, subformDiv) {
    return 'refresh_subform/' + dynamicPage.dynamic_gid + '/' + currentElem.value;
}

function displayDocForModel(model){
    model_id = model.find(":selected").val();
    $("#data_model_" + model_id.replace(/ /g,"_")).css("display", "block");
}

function fillDoc(){
    var model = $('#model');
    model.on("change", function(event){
        $('div[id^="data_model_"]').css("display", "none");
        displayDocForModel(model);
    });

    displayDocForModel(model);
}