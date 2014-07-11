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
    sliderDefaultState : {},
    treeState: {},
    dynamic_gid: null
};

dynamicPage._initSliders = function(){
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
                dynamicPage.onParameterChanged(option.name, target.value);
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

    for (var i = 0; i < dynamicPage.sliderDefaultState.length; i++){
        var option = dynamicPage.sliderDefaultState[i];
        var slider = $("#slider_" + option.name);
        var input = $("#value_" + option.name);

        _initSlider(slider, input, option);

    }
};

dynamicPage.onSliderReset = function(){
    for (var i = 0; i < dynamicPage.sliderDefaultState.length; i++) {
        var option = dynamicPage.sliderDefaultState[i];
        var slider = $("#slider_" + option.name);
        slider.slider('value', option.default);
    }
};

dynamicPage._url = function(func, tail){
    var url = '/burst/create_dynamic/' + func + '/' + dynamicPage.dynamic_gid;
    if (tail != null){
        url+= '/' + tail;
    }
    return url;
};

dynamicPage._fetchSlidersFromServer = function(){
    var sliderContainer = $('#div_spatial_model_params');
    sliderContainer.empty();

    doAjaxCall({
        url: dynamicPage._url('sliders_fragment'),
        success: function(fragment) {
            sliderContainer.html(fragment);
            $('#reset_sliders').click(dynamicPage.onSliderReset);
            MathJax.Hub.Queue(["Typeset", MathJax.Hub, 'div_spatial_model_params']);
            dynamicPage._initSliders();
            setupMenuEvents(sliderContainer);
        }
    });
};

dynamicPage.onModelChanged = function(name){
    doAjaxCall({
        url: dynamicPage._url('model_changed', name) ,
        success: function(data){
            dynamicPage.sliderDefaultState = $.parseJSON(data).options;
            dynamicPage._fetchSlidersFromServer();
        }
    });
};

dynamicPage.onParameterChanged = function(name, value){
    doAjaxCall({
        url: dynamicPage._url('parameter_changed'),
        data: {'name' : name, 'value': value},
        success: function(data){

        }
    });
};

dynamicPage.onSubmit = function(event){
    var name = $('#dynamic_name').val().trim();
    if (name.length ) {
        doAjaxCall({
            url: dynamicPage._url('submit', name),
            success: function(){
                displayMessage('Dynamic saved', 'importantMessage');
            }
        });
    }else{
        displayMessage('A name is required', 'errorMessage');
    }
    event.preventDefault();
};

dynamicPage.onIntegratorChanged = function(state){
    doAjaxCall({
        url: dynamicPage._url('integrator_changed'),
        data: state
    });
};

dynamicPage.debouncedOnIntegratorChanged = $.debounce( 250, dynamicPage.onIntegratorChanged);

// Detect changes by doing a tree diff. This diff is simple but relies on a specific tree structure.
dynamicPage.onTreeChange = function(){
    var state = getSubmitableData('left_input_tree');
    var previous = dynamicPage.treeState;
    if (state.model_type !== previous.model_type){
        dynamicPage.onModelChanged(state.model_type);
    }else if (state.dynamic_name === previous.dynamic_name){
        // Name has not changed. The change is in the integrator subtree
        dynamicPage.debouncedOnIntegratorChanged(state);
    }
    dynamicPage.treeState = state;
};

dynamicPage.main = function(dynamic_gid){
    dynamicPage.dynamic_gid = dynamic_gid;
    $('.field-adapters').hide(); // hide expand range buttons. Careful as this class is used for other things as well
    // listen for changes of the left tree
    $('#left_input_tree').find('input').add('select').change(dynamicPage.onTreeChange);
    $('#base_spatio_temporal_form').submit(dynamicPage.onSubmit);
    dynamicPage.onTreeChange();
};


