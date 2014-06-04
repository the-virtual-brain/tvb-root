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
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0
 *
 **/

// -------------------------------------------------------------
//              Channel selector methods start from here
// -------------------------------------------------------------

/*
 * Add all channels to the channelsArray list and make sure that
 * all of the checkbox are checked.
 */
function checkAll(entryArray, channelsArray) {

	var divId = 'table_' + entryArray;
    $('div[id^="'+ divId + '"] input').each(function () {
        if (this.type == "checkbox") {
            var channelId = parseInt(this.id.split("channelChk_")[1]);
            var elemIdx = $.inArray(channelId, channelsArray);
		    if (elemIdx == -1) {
		        channelsArray.push(channelId);
		    }
		    $("#channelChk_" + channelId).attr('checked', true);
        }
    });
}

/*
 * Remove all channels to the channelsArray list and make sure that
 * none of the checkbox are checked.
 */
function clearAll(entryArray, channelsArray) {

	var divId = 'table_' + entryArray;
    $('div[id^="'+ divId + '"] input').each(function () {
        if (this.type == "checkbox") {
            var channelId = this.id.split("channelChk_")[1];
            var elemIdx = $.inArray(parseInt(channelId), channelsArray);
		    if (elemIdx != -1) {
		        channelsArray.splice(elemIdx, 1);
		    }
        }
        $("#channelChk_" + channelId).attr('checked', false);
    });
}

/*
 * Update the channelsArray array whenever a checkbox is clicked.
 */
function updateChannelsList(domElem, channelsArray) {

    var channelId = parseInt(domElem.id.split("channelChk_")[1]);
    var elemIdx;
    if (domElem.checked) {
        elemIdx = $.inArray(channelId, channelsArray);
	    if (elemIdx == -1) {
	        channelsArray.push(channelId);
	    }
    } else {
        elemIdx = $.inArray(channelId, channelsArray);
	    if (elemIdx != -1) {
	        channelsArray.splice(elemIdx, 1);
	    }
    }
}

// -------------------------------------------------------------
//              Channel selector methods end here
// -------------------------------------------------------------


// -------------------------------------------------------------
//              Datatype methods mappings start from here
// -------------------------------------------------------------

function readDataPageURL(baseDatatypeMethodURL, fromIdx, toIdx, stateVariable, mode, step) {
	if (stateVariable == undefined || stateVariable == null) {
		stateVariable = 0;
	}
	if (mode == undefined || stateVariable == null) {
		mode = 0;
	}
	if (step == undefined || step == null) {
		step = 1;
	}
	return baseDatatypeMethodURL + '/read_data_page/False?from_idx=' + fromIdx +";to_idx=" + toIdx + ";step=" + step + ";specific_slices=[null," + stateVariable + ",null," + mode +"]";
}

function readDataChannelURL(baseDatatypeMethodURL, fromIdx, toIdx, stateVariable, mode, step, channels) {
	var baseURL = readDataPageURL(baseDatatypeMethodURL, fromIdx, toIdx, stateVariable, mode, step);
	return baseURL.replace('read_data_page', 'read_channels_page') + ';channels_list=' + channels;
}
 
// -------------------------------------------------------------
//              Datatype methods mappings end here
// -------------------------------------------------------------