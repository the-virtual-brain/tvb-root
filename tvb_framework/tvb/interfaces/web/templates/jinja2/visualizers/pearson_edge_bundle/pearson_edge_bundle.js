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
 * .. moduleauthor:: Vlad Farcas <vlad.farcas@codemart.ro>
 **/


var PearsonEdgesData = {
    region_labels: [""],
    matrix: [],
    svg: {
        d3: null,
        svg: null
    },
    url_base: "",
    mode: "",
    mode_list: "",
    state: "",
    state_list: "",
    thresh: 0,
};

function PE_InitChord(url_base, labels, state, mode, thresh) {

    PearsonEdgesData.region_labels = labels;
    PearsonEdgesData.state = state;
    PearsonEdgesData.mode = mode;
    PearsonEdgesData.svg.d3 = d3.select("#middle-edge-bundle");
    PearsonEdgesData.svg.svg = $("#middle-edge-bundle");
    PearsonEdgesData.url_base = url_base;
    PearsonEdgesData.thresh = thresh;

    _PE_Ajaxify();
}

function PE_TriggerRedraw() {

    PearsonEdgesData.mode = ($("#mode_select").find("option:selected").attr("value"));
    PearsonEdgesData.state = ($("#state_select").find("option:selected").attr("value"));
    _PE_Ajaxify();
}

//threshold of which values in the matrix to be considered edges for display
function PE_ChangeThreshold(newVal) {

    $("#slider").attr("value", newVal);
    $("#valBox").html(newVal);
    $("#slider-value").html(newVal);
    PearsonEdgesData.thresh = parseFloat(newVal);
    _PE_RefreshEdges();
}


function _PE_Ajaxify() {

    doAjaxCall({
        url: PearsonEdgesData.url_base + "selected_state=" + PearsonEdgesData.state + ";selected_mode=" + PearsonEdgesData.mode,
        type: 'POST',
        async: true,
        success: function (data) {
            PearsonEdgesData.matrix = $.parseJSON(data);
            _PE_RefreshEdges();
        }
    });
}

function _PE_RefreshEdges() {
    PearsonEdgesData.svg.d3.selectAll("*").transition().duration(100).style("fill-opacity", "0");
    PearsonEdgesData.svg.d3.selectAll("*").remove();
    PearsonEdgesData.svg.d3.selectAll("*").transition().duration(100).style("fill-opacity", "1");

    HEB_InitData(PearsonEdgesData, function (d) {
        return d >= PearsonEdgesData.thresh;
    });
}