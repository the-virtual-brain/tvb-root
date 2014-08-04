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

/* globals doAjaxCall, displayMessage */

//Used since calendars need an input field with an id.
var nextId = 0;

function sortOnKeys(dict) {

    var sorted = [];
    for (var key in dict) {
        sorted[sorted.length] = key;
    }
    sorted.sort();

    var tempDict = {};
    for (var i = 0; i < sorted.length; i++) {
        tempDict[sorted[i]] = dict[sorted[i]];
    }

    return tempDict;
}

function addFilter(div_id, filters) {
    var div = document.getElementById(div_id);
    //Create a new div for the filter
    var newDiv = document.createElement("div");
    //Create a label
    var label = document.createElement("label");
    var text = document.createTextNode("Filter : ");
    label.appendChild(text);

    //This will be the operation select item
    var operation = document.createElement("select");

    //This will be the select row to filter by
    var filter = document.createElement("select");
    //On change update the possible operations

    var pos = 0;
    //Is it a date field or not. So far validation is if 'date' appears in name
    //To be replaced with an 'expected type' received from the controllers
    var isDate = 0;
    //Order dictionary, to always keep the same display order.
    filters = sortOnKeys(filters);
    //Iterate over the filters dictionary and create the possible values
    for (var i in filters) {
        filter.options[pos] = new Option(filters[i]['display'], i);
        //Fill default options for the operation select
        if (pos == 0) {
            var available_ops = filters[i]['operations'];
            for (var j = 0; j < available_ops.length; j++) {
                operation.options[j] = new Option(available_ops[j], available_ops[j]);
            }
            if (filters[i]['type'] == 'date') {
                isDate = 1;
            }
        }
        pos++;
    }
    //Input field
    var input = document.createElement("input");
    input.type = "text";
    input.id = "calendar" + nextId;
    nextId++;
    input.name = "values";

    if (isDate == 1) {
        input.readOnly = true;
        var calendarImg = new Image();
        calendarImg.src = "/static/style/img/calendar.png";
        calendarImg.onclick = function () {
            NewCssCal(input.id);
        };
    }


    filter.onchange = function () {
        operation.options.length = 0;
        //Get new operations to display
        var newOptions = filters[this.value]['operations'];
        for (var j = 0; j < newOptions.length; j++) {
            operation.options[j] = new Option(newOptions[j], newOptions[j]);
        }
        //Now check if date is needed or not. If not remove from parent.
        var div_parent = this.parentNode.parentNode;
        var siblings = this.parentNode.childNodes;
        var inputWithId = null;
        var calendar = null;
        for (var j = 0; j < siblings.length; j++) {
            if (siblings[j].id != '' && siblings[j].id != null) {
                inputWithId = siblings[j];
            }
            if (siblings[j].tagName == 'IMG') {
                calendar = siblings[j];
            }
        }
        if (filters[this.value]['type'] != "date") {
            this.parentNode.removeChild(calendar);
            input.readOnly = false;
        }
        else {
            //If calendar is needed but was deleted, create a new one.
            if (calendar == null) {
                input.readOnly = true;
                var calendarImg = new Image();
                calendarImg.src = "/static/style/img/calendar.png";
                calendarImg.onclick = function () {
                    NewCssCal(inputWithId.id);
                };
                this.parentNode.insertBefore(calendarImg, inputWithId.nextSibling);
            }
        }
        $(input).val("");
    };
    //Remove button
    var button = document.createElement("input");
    button.type = "button";
    button.value = " Drop Filter ";
    button.onclick = function () {
        div.removeChild(newDiv);
    };

    //Add all components to div
    newDiv.insertBefore(button, newDiv.firstChild);
    if (isDate == 1) {
        newDiv.insertBefore(calendarImg, newDiv.firstChild);
    }
    newDiv.insertBefore(input, newDiv.firstChild);
    newDiv.insertBefore(operation, newDiv.firstChild);
    newDiv.insertBefore(filter, newDiv.firstChild);
    newDiv.insertBefore(label, newDiv.firstChild);
    div.appendChild(newDiv);
}


function refreshData(parentDivId, divSufix, name, sessionStoredTreeKey, gatheredData) {
    var divId = parentDivId + divSufix;
    if (!gatheredData) {
        //Thiw will gather all the data from the filters and make an
        //ajax request to get new data
        var filterDiv = document.getElementById(divId);
        var children = filterDiv.childNodes;
        var fields = [];
        var operations = [];
        var values = [];
        //First gather all the information
        for (var i = 0; i < children.length; i++) {
            //There is a new div for each filter.
            if (children[i].tagName == "DIV") {
                var informations = children[i].children;
                //Get info about the filters.
                if (informations[3].value.trim().length > 0) {
                    fields.push(informations[1].value);
                    operations.push(informations[2].value);
                    values.push(informations[3].value.trim());
                    displayMessage("Filters processed");
                } else {
                    displayMessage("Please set a value for all the filters.", "errorMessage");
                    return;
                }
            }
        }
        if (fields.length === 0 && operations.length === 0 && values.length === 0) {
            displayMessage("Cleared filters");
        }

        gatheredData = {
            'fields': fields,
            'operations': operations,
            'values': values
        };
    }
    var elements = document.getElementsByName(name);
    if (elements.length < 1) {
        // Hidden elements from simulator interface or missing parameters will be ignored.
        displayMessage("Filter could not be applied! " + name, "infoMessage");
        return;
    }
    //Make a request to get new data
    doAjaxCall({ async: false,  //todo: Is this sync really needed? It slows down the page.
        type: 'GET',
        url: "/flow/getfiltereddatatypes/" + name + "/ " + parentDivId + '/' + sessionStoredTreeKey + '/' + $.toJSON(gatheredData),
        success: function (r) {
            var elements = document.getElementsByName(name);
            //Look for the previous select input whose data needs to be refreshed
            var parentDiv;
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
                        parentDiv = elements[i].parentNode;
                        replaceSelect(parentDiv, r, name);
                    }
                }
            } else if (elements.length === 1) {
                parentDiv = elements[0].parentNode;
                replaceSelect(parentDiv, r, name);
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
        var filterData = {'fields': [filterField],
            'operations': ["in"],
            'values': [currentSelectedGID.split(' ')]};


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
                parentDivID += option_name;
                refreshData(parentDivID, linkedInputName + 'data_select', linkedInputName, treeSessionKey, filterData);
            });
        }
    }
}
