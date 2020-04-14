/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and 
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

var RANGE_PARAMETER_1 = "range_1";
var RANGE_PARAMETER_2 = "range_2";
var BUTTON_EXPAND_SUFFIX = "_buttonExpand";

function disableRangeComponent(containerTableId, inputName) {
    var first_ranger = document.getElementById(RANGE_PARAMETER_1);
    var second_ranger = document.getElementById(RANGE_PARAMETER_2);
    if (first_ranger.value == inputName){
        first_ranger.value = '0';
    } else if (second_ranger.value == inputName){
        second_ranger.value = '0';
    }

    var topTable = $('#' + containerTableId);
    topTable.hide();
    // Disable all input fields in the ranger
    topTable.find('input').attr('disabled', 'disabled');

    $("#" + inputName).removeAttr('disabled').css('display', 'block');
    $('#' + containerTableId + BUTTON_EXPAND_SUFFIX).val('Expand Range');
    $("#" + containerTableId).find('input[type=number]').off('blur');
    //RANGE_computeNrOfOps();
}

// this script is loaded dynamically. The comment below is a source map. This is to see the file in js tools
//@ sourceURL=range.js