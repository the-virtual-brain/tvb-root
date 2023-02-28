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

/**
 * Created by vlad.farcas on 7/21/2017.
 *
 * Many thanks to Mike Bostock's bl.ock: https://bl.ocks.org/mbostock/7607999
 *
 * connectivity_edge_bundle viewer
 *
 */

var ConnectivityEdgesData = {
    region_labels: [""],
    matrix: [],
    svg: {
        d3: null,
        svg: null,
    },
    data_url: "",
    state: "weights"
};

function _CE_Ajaxify() {

    doAjaxCall({
        url: ConnectivityEdgesData.data_url,
        type: 'POST',
        async: true,
        success: function (data) {
            ConnectivityEdgesData.matrix = $.parseJSON(data);
            HEB_InitData(ConnectivityEdgesData, function (d) {
                return d !== 0;
            });
        }
    });
}


function CE_InitChord(data_url, labels) {

    ConnectivityEdgesData.region_labels = labels;
    ConnectivityEdgesData.svg.d3 = d3.select("#middle-edge-bundle");
    ConnectivityEdgesData.svg.svg = $("#middle-edge-bundle");
    ConnectivityEdgesData.data_url = data_url;

    _CE_Ajaxify();
}


function CE_UpdateMatrix() {

    ConnectivityEdgesData.svg.d3.selectAll("*").transition().duration(100).style("fill-opacity", "0");
    ConnectivityEdgesData.svg.d3.selectAll("*").remove();

    // Update data URL and retrieve data from
    let newstate = ConnectivityEdgesData.state === "weights" ? "tract_lengths" : "weights";
    ConnectivityEdgesData.data_url = ConnectivityEdgesData.data_url.replace(ConnectivityEdgesData.state, newstate);
    ConnectivityEdgesData.state = newstate;
    _CE_Ajaxify();

    ConnectivityEdgesData.svg.d3.selectAll("*").transition().duration(100).style("fill-opacity", "1");
}