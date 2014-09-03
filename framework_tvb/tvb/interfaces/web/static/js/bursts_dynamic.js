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

(function(){

function initSliderGroup (states, onChange){
    function _initSlider(slider, input, option){
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
                onChange(option.name, target.value);
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
    }

    for (var i = 0; i < states.length; i++){
        var option = states[i];
        var slider = $("#slider_" + option.name);
        var input = $("#value_" + option.name);
        _initSlider(slider, input, option);
    }
}

function resetSliderGroup(states){
    for (var i = 0; i < states.length; i++) {
        var option = states[i];
        var slider = $("#slider_" + option.name);
        slider.slider('value', option.default);
    }
}

function getSliderGroupValues(states) {
    var name2val = {};
    for (var i = 0; i < states.length; i++) {
        var option = states[i];
        var slider = $("#slider_" + option.name);
        name2val[option.name] = slider.slider('value');
    }
    return name2val;
}

function _initAxisSlider(sel, span_sel, opt){
    function update_span(r){
        $(span_sel).text(r[0] + ' .. ' + r[1]);
    }
    $(sel).slider({
        range:true,
        min: opt.min,
        max: opt.max, values:[opt.lo, opt.hi],
        step: (opt.max - opt.min)/1000,
        slide: function(ev, target) {
            update_span(target.values);
        },
        change : function(ev, target){
            update_span(target.values);
            onGraphChanged();
        }
    });
}

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
            $('#reset_sliders').click(function(){
                resetSliderGroup(dynamicPage.paramDefaults);
            });
            $('#reset_state_variables').click(function(){
                resetSliderGroup(dynamicPage.graphDefaults.state_variables);
            });
            $('#reset_axes').click(function() {
                //not nice
                var opt = dynamicPage.graphDefaults.x_axis;
                $('#slider_x_axis').slider({ min: opt.min, max: opt.max, values: [opt.lo, opt.hi] });
                opt = dynamicPage.graphDefaults.y_axis;
                $('#slider_y_axis').slider({ min: opt.min, max: opt.max, values: [opt.lo, opt.hi] });
                //reset mode and state var selection as well
                $('#mode').val(0).change(); // change events do not fire when select's are changed by val()
                $('#svx').val(dynamicPage.graphDefaults.state_variables[0].name).change();
                $('#svy').val(dynamicPage.graphDefaults.state_variables[1].name).change();
            });
            MathJax.Hub.Queue(["Typeset", MathJax.Hub, 'div_spatial_model_params']);
            initSliderGroup(dynamicPage.paramDefaults, onParameterChanged);
            initSliderGroup(dynamicPage.graphDefaults.state_variables, onGraphChanged);
            $('#mode').change(onGraphChanged);
            $('#svx').change(onGraphChanged);
            $('#svy').change(onGraphChanged);
            _initAxisSlider('#slider_x_axis', '#x_range_span', dynamicPage.graphDefaults.x_axis);
            _initAxisSlider('#slider_y_axis', '#y_range_span', dynamicPage.graphDefaults.y_axis);
            setupMenuEvents(sliderContainer);
        }
    });
}

function onModelChanged(name){
    doAjaxCall({
        showBlockerOverlay: true,
        url: _url('model_changed', name) ,
        success: function(data){
            data = JSON.parse(data);
            dynamicPage.paramDefaults = data.params;
            dynamicPage.graphDefaults = data.graph_params;
            _fetchSlidersFromServer();
        }
    });
}

function _onParameterChanged(){
    doAjaxCall({
        showBlockerOverlay: true,
        url: _url('parameters_changed'),
        data: {params: JSON.stringify(getSliderGroupValues(dynamicPage.paramDefaults))}
    });
}

// Resetting a slider group will trigger change events for each slider. The handler does a slow ajax so debounce the handler
var onParameterChanged = $.debounce(50, _onParameterChanged);

function _onGraphChanged(){
    var graph_state = {
        mode: $('#mode').val(),
        svx: $('#svx').val(),
        svy: $('#svy').val(),
        state_vars: getSliderGroupValues(dynamicPage.graphDefaults.state_variables),
        x_range: $('#slider_x_axis').slider('values'),
        y_range: $('#slider_y_axis').slider('values')
    };

    doAjaxCall({
        showBlockerOverlay: true,
        url: _url('graph_changed'),
        data: { graph_state: JSON.stringify(graph_state)}
    });
}

// see onParameterChanged
var onGraphChanged = $.debounce(50, _onGraphChanged);

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
        showBlockerOverlay: true,
        url: _url('integrator_changed'),
        data: state
    });
}

// Event debouncing makes less sense now that the requests have been made blocking.
var debouncedOnIntegratorChanged = $.debounce( 50, onIntegratorChanged);

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
}

dynamicPage.main = main;

})();
