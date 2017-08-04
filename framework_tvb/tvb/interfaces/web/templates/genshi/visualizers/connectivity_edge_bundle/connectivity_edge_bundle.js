/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

var ChordData = {
    region_labels: [""],
    matrix: [],
    svg: {
        d3: null,
        svg: null,
    },
    url_base: "",
    state: "tract_lengths"
};

function ajaxify() {

    let newstate = ChordData.state === "weights" ? "tract_lengths" : "weights";

    doAjaxCall({
        url: ChordData.url_base.replace(ChordData.state, newstate),
        type: 'POST',
        async: true,
        success: function (data) {
            ChordData.matrix = $.parseJSON(data);
            init_data(ChordData, function (d) {
                return d !== 0;
            });
        }
    });

    ChordData.state = newstate;
}

function init_chord(url_base, labels) {

    ChordData.region_labels = labels;
    ChordData.svg.d3 = d3.select("#middle-edge-bundle");
    ChordData.svg.svg = $("#middle-edge-bundle");
    ChordData.url_base = url_base;

    //add event listener to switch button
    $("#switch-matrix").on("click", function (e) {

        ChordData.svg.d3.selectAll("*").transition().duration(100).style("fill-opacity", "0");
        ChordData.svg.d3.selectAll("*").remove();

        ajaxify();

        ChordData.svg.d3.selectAll("*").transition().duration(100).style("fill-opacity", "1");
    });

    ajaxify();
}