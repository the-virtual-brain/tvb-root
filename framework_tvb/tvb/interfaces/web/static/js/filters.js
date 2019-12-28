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

/* globals doAjaxCall, displayMessage */


/**
 * Creates the operation select, the input and the drop button
 */
function _FIL_createUiForFilterType(filter, newDiv, isDate){
    var operation = $('<select></select>');
    for (var j = 0; j < filter.operations.length; j++) {
        var op = filter.operations[j];
        operation.append(new Option(op, op));
    }
    var input = $('<input type="text" name="values"/>');
    var dropButton = $('<input type="button" value=" Drop Filter "/>');
    newDiv.append(operation, input, dropButton);

    dropButton.click(function () {
        newDiv.remove();
    });

    if (isDate){
        input.datepicker({
            changeMonth: true,
            changeYear: true,
            showOn: "both",
            buttonImage: "/static/style/img/calendar.png",
            buttonImageOnly: true,
            dateFormat : "mm-dd-yy",
            buttonText: "pick a date"
        });
    }
}

function addFilter(div_id, filters) {
    //Create a new div for the filter
    var newDiv = $('<div> <label> Filter : </label> </div>');
    $('#' + div_id).append(newDiv);

    //This will be the select row to filter by
    var filter = $('<select></select>');

    //Iterate over the filters dictionary and create the possible values
    var filter_names = Object.keys(filters).sort();
    for (var i = 0; i < filter_names.length; i++) {
        var k = filter_names[i];
        filter.append( new Option(filters[k].display, k));
    }

    newDiv.append(filter);
    var first_filter = filters[filter_names[0]];
    _FIL_createUiForFilterType(first_filter, newDiv, first_filter.type === 'date');

    filter.change(function () {
        // remove all nodes to the right of filter
        var children = newDiv.children();
        for (var i = 2; i < children.length; i++){
            $(children[i]).remove();
        }
        // recreate them
        _FIL_createUiForFilterType(filters[this.value], newDiv, filters[this.value].type === 'date');
    });
}

/** gather all the data from the filters */
function _FIL_gatherData(divId){
    var children = $('#'+divId).children('div');
    var fields = [];
    var operations = [];
    var values = [];

    for (var i = 0; i < children.length; i++) {
        var elem = children[i].children;
        //Get info about the filters.
        if (elem[3].value.trim().length > 0) {
            fields.push(elem[1].value);
            operations.push(elem[2].value);
            values.push(elem[3].value.trim());
            displayMessage("Filters processed");
        } else {
            displayMessage("Please set a value for all the filters.", "errorMessage");
            return;
        }
    }
    if (fields.length === 0 && operations.length === 0 && values.length === 0) {
        displayMessage("Cleared filters");
    }

    return { fields: fields, operations: operations, values: values};
}

function refreshData(parentDivId, divSufix, name, sessionStoredTreeKey, gatheredData) {
    var divId = parentDivId + divSufix;
    if (!gatheredData) {
        //gather all the data from the filters and make an
        //ajax request to get new data
        gatheredData = _FIL_gatherData(divId);
        if (gatheredData == null) {
            return;
        }
    }

    if (document.getElementsByName(name).length < 1) {
        // Hidden elements from simulator interface or missing parameters will be ignored.
        displayMessage("Filter could not be applied! " + name, "infoMessage");
        return;
    }

    // This argument is required by the server.
    // If absent set a falsy default as updateDivContent() checks for it. Has to be a string because it's in the url.
    if (parentDivId === ""){
        parentDivId = " ";
    }
    //Make a request to get new data
    doAjaxCall({
        async: false,  //todo: Is this sync really needed? It slows down the page.
        type: 'GET', //todo: why is this a get? a post with the json seems better.
        url: "/flow/getfiltereddatatypes/" + name + "/" + parentDivId + '/' + sessionStoredTreeKey + '/' + $.toJSON(gatheredData),
        success: function (r) {
            var elements = document.getElementsByName(name);
            //Look for the previous select input whose data needs to be refreshed
            if (elements.length > 1) {
                //If more than one was found it's because of multiple algorithms
                //We need to get only the one corresponding to the current algorithm
                for (var i = 0; i < elements.length; i++) {
                    var parent = elements[i].parentNode;
                    //In case more components with the same name exist look for the
                    //parent's div id
                    while (parent.id == '' || parent.id == null) {
                        parent = parent.parentNode;
                    }
                    if (divId != null && divId.indexOf(parent.id) !== -1) {
                        //Remove the childs from this div and then recreate the components
                        //using the html returned by the ajax call
                        replaceSelect(elements[i].parentNode, r, name);
                    }
                }
            } else if (elements.length === 1) {
                replaceSelect(elements[0].parentNode, r, name);
            } else {
                displayMessage("Filter could not be applied!" + name, "infoMessage");
                return;
            }
            displayMessage("Filters applied...");
        },
        error: function (r) {
            displayMessage("Invalid filter data.", "errorMessage");
        }
    });
}


/**
 * After the user executes a filter than we have to replace the select with the old option with
 * the select which contains only the options that satisfies the filters.
 *
 * @param parentDiv the parent div in which is located the select
 * @param newSelect the html that contains the new select
 * @param selectName the name of the old select
 */
function replaceSelect(parentDiv, newSelect, selectName) {
    var allChildren = parentDiv.children;
    for (var j = 0; j < allChildren.length; j++) {
        if (allChildren[j].nodeName == 'SELECT' && allChildren[j].name == selectName) {
            $(newSelect).insertAfter($(allChildren[j]));
            parentDiv.removeChild(allChildren[j]);
            break;
        }
    }
}

/**
 * Filter fields which are linked with current entity.
 * @param {list} linkedDataList list of lists.
 * @param {string} currentSelectedGID for current input
 * @param {string} treeSessionKey Key
 */
function filterLinked(linkedDataList, currentSelectedGID, treeSessionKey) {
    if (currentSelectedGID.length < 1) {
        return;
    }
    for (var i = 0; i < linkedDataList.length; i++) {
        var linkedData = linkedDataList[i];
        var elemName = linkedData.linked_elem_name;

        var filterField = linkedData.linked_elem_field;
        var filterData = {
            'fields': [filterField],
            'operations': ["in"],
            'values': [currentSelectedGID.split(' ')]
        };

        if (!linkedData.linked_elem_parent_name && !linkedData.linked_elem_parent_option) {
            refreshData("", elemName + 'data_select', elemName, treeSessionKey, filterData);
        }

        var linkedInputName = linkedData.linked_elem_parent_name + "_parameters_option_";
        var parentDivID = 'data_' + linkedData.linked_elem_parent_name;

        if (linkedData.linked_elem_parent_option) {
            linkedInputName = linkedInputName + linkedData.linked_elem_parent_option + "_" + elemName;
            parentDivID += linkedData.linked_elem_parent_option;
            refreshData(parentDivID, linkedInputName + 'data_select', linkedInputName, treeSessionKey, filterData);
        } else {
            $("select[id^='" + linkedInputName + "']").each(function () {
                if ($(this)[0].id.indexOf("_" + elemName) < 0) {
                    return;
                }
                var option_name = $(this)[0].id.replace("_" + elemName, '').replace(linkedInputName, '');
                linkedInputName = $(this)[0].id;
                parentDivID += option_name; // todo : possible bug. option names will be concatenated many times if this each runs more than once
                refreshData(parentDivID, linkedInputName + 'data_select', linkedInputName, treeSessionKey, filterData);
            });
        }
    }
}
