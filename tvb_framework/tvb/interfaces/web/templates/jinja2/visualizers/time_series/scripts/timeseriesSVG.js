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
 **/
/* global tv, d3, TVBUI */

var TS_SVG_selectedChannels = [];
// Store a list with all the channel labels so one can easily refresh between them on channel refresh
var allChannelLabels = [];

var MAX_INITIAL_CHANNELS = 20;
var _initial_magic_fcs_amp_scl;
var tsView;

/**
 * Initialize selected channels to default values. This is used when no selection ui is present (portlet mode)
 * initializes global TS_SVG_selectedChannels
 * Takes the first 20 channels or all
 */
function _initDefaultSelection(){
    for(var i = 0; i < Math.min(MAX_INITIAL_CHANNELS, allChannelLabels.length); i++){
            TS_SVG_selectedChannels.push(i);
    }
}

/**
 * Initializes selected channels
 * initializes global TS_SVG_selectedChannels
 */
function _initSelection(filterGid) {
    // initialize selection component
    var regionSelector = TVBUI.regionSelector("#channelSelector", {filterGid: filterGid});
    var initialSelection = regionSelector.val();

    if (initialSelection.length == 0) {
        // If there is no selection use the default
        _initDefaultSelection();
        regionSelector.val(TS_SVG_selectedChannels);
    } else if (initialSelection.length > MAX_INITIAL_CHANNELS) {
        // Take a default of maximum 20 channels at start to be displayed
        TS_SVG_selectedChannels = initialSelection.slice(0, MAX_INITIAL_CHANNELS);
        regionSelector.val(TS_SVG_selectedChannels);
    } else {
        // to not invoke val to keep the named selection in the dropdown
        TS_SVG_selectedChannels = initialSelection;
    }
    // we bind the event here after the previous val calls
    // do not want those to trigger the change event as tsView is not initialized yet
    regionSelector.change(function (value) {
        TS_SVG_selectedChannels = [];
        for (var i = 0; i < value.length; i++) {
            TS_SVG_selectedChannels.push(parseInt(value[i], 10));
        }
        refreshChannels();
    });
    var modeSelector = TVBUI.modeAndStateSelector("#channelSelector", 0);
    modeSelector.modeChanged(_changeMode);
    modeSelector.stateVariableChanged(_changeStateVariable);
}

function _compute_labels_for_current_selection() {
    var selectedLabels = [];
    for(var i = 0; i < TS_SVG_selectedChannels.length; i++){
        selectedLabels.push(allChannelLabels[TS_SVG_selectedChannels[i]]);
    }
    return selectedLabels;
}

function _updateScalingFromSlider(value){
    if (value == null){
        value = $("#ctrl-input-scale").slider("value");
    }
    var expo_scale = (value - 50) / 50; // [1 .. -1]
    var scale = Math.pow(10, expo_scale*4); // [1000..-1000]
    tsView.magic_fcs_amp_scl = _initial_magic_fcs_amp_scl * scale;
    tsView.prepare_data();
    tsView.render_focus();
    if(scale >= 1){
        $("#display-scale").html("1 * " + scale.toFixed(2));
    }else{
        $("#display-scale").html("1 / " + (1/scale).toFixed(2));
    }
}
/*
 * Do any required initializations in order to start the viewer.
 *
 * @param baseURL: the base URL from tvb in order to call datatype methods
 * @param isPreview: boolean that tell if we are in burst page preview mode or full viewer
 * @param dataShape: the shape of the input timeseries
 * @param t0: starting time
 * @param dt: time increment
 * @param channelLabels: a list with the labels for all the channels
 */
function initTimeseriesViewer(baseURL, isPreview, dataShape, t0, dt, channelLabels, filterGid) {

    // Store the list with all the labels since we need it on channel selection refresh
    allChannelLabels = channelLabels;

    // setup dimensions, div, svg elements and plotter
    var ts = tv.plot.time_series();

    isPreview = (isPreview === "True");

    if(isPreview){
        _initDefaultSelection();
    }else{
        _initSelection(filterGid);
    }

    dataShape = $.parseJSON(dataShape);
    dataShape[2] = TS_SVG_selectedChannels.length;

    // configure data
    ts.baseURL(baseURL).preview(isPreview).mode(0).state_var(0);
    ts.shape(dataShape).t0(t0).dt(dt);
    ts.labels(_compute_labels_for_current_selection());
    ts.channels(TS_SVG_selectedChannels);
    // run
    resizeToFillParent(ts);
    ts(d3.select("#time-series-viewer"));
    tsView = ts;

    // This is arbitrarily set to a value. To be consistent with tsview we rescale relative to this value
    _initial_magic_fcs_amp_scl = tsView.magic_fcs_amp_scl;

    if (!isPreview) {
        $("#ctrl-input-scale").slider({ value: 50, min: 0, max: 100,
            slide: function (event, target) {
                _updateScalingFromSlider(target.value);
            }
        });
    }
}

function resizeToFillParent(ts){
    var container, width, height;

    if(!ts.preview()) {
        container = $('#time-series-viewer').parent();
        width = container.width();
        height = container.height() - 80;
    }else{
        container = $('body');
        width = container.width();
        height = container.height() - 10;
    }
    ts.w(width).h(height);
}

/*
 * Get required data for the channels in TS_SVG_selectedChannels. If none
 * exist then just use the previous 'displayedChannels' (or default in case of first run).
 */
function refreshChannels() {

    var selectedLabels = [];
    var shape = tsView.shape();

    if (TS_SVG_selectedChannels.length === 0) {
        // if all channels are deselected show them all
        selectedLabels = allChannelLabels;
        shape[2] = allChannelLabels.length;
    } else {
        for (var i = 0; i < TS_SVG_selectedChannels.length; i++) {
            selectedLabels.push(allChannelLabels[TS_SVG_selectedChannels[i]]);
        }
        shape[2] = TS_SVG_selectedChannels.length;
    }

    var new_ts = tv.plot.time_series();

    // configure data
    new_ts.baseURL(tsView.baseURL()).preview(tsView.preview()).mode(tsView.mode()).state_var(tsView.state_var());
    new_ts.shape(shape).t0(tsView.t0()).dt(tsView.dt());
    new_ts.labels(selectedLabels);
    // Usually the svg component shows the channels stored in TS_SVG_selectedChannels
    // and that variable is in sync with the selection component.
    // But if the selection is empty and we show a timeSeriesSurface
    // then new_ts will get time series for all 65k vertices from the server.
    if (TS_SVG_selectedChannels.length !== 0){
        new_ts.channels(TS_SVG_selectedChannels);
    }else {
        new_ts.channels(tv.ndar.range(allChannelLabels.length).data);
    }

    resizeToFillParent(new_ts);
    $('#time-series-viewer').empty();
    new_ts(d3.select("#time-series-viewer"));
    tsView = new_ts;
    // The new_ts(...) call above will trigger a data load from the server based on the selected channels
    // Until that internal ajax returns the new_ts has no time series data
    // _updateScalingFromSlider calls new_ts render so that the rendering takes into account the slider value
    // We use settimeout to defer _updateScalingFromSlider until data is available to it
    // todo: The proper way to handle this is to subscribe to a data_loaded event raised by new_ts
    function await_data() {
        if (tsView.ts() == null) {
            setTimeout(await_data, 100);
        } else {
            _updateScalingFromSlider();
        }
    }
    await_data();
}

function _changeMode(id, val) {
    tsView.mode(parseInt(val));
    refreshChannels();
}

function _changeStateVariable(id, val) {
    tsView.state_var(parseInt(val));
    refreshChannels();
}
