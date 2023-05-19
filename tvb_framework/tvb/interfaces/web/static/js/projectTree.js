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
 * Project Data Structure functions are here.
 */
var TREE_lastSelectedNode = undefined;
var TREE_lastSelectedNodeType = undefined;

//-----------------------------------------------------------------------
//                   TREE Section starts here
//-----------------------------------------------------------------------

function select_tree_node(treeId) {
        var treeElem = $("#" + treeId);
        treeElem.jstree("deselect_all");
        TVB_skipDisplayOverlay = true;
        if (TREE_lastSelectedNode != undefined && TREE_lastSelectedNodeType != undefined && TREE_lastSelectedNodeType == TVB_NODE_DATATYPE_TYPE) {
            treeElem.jstree("select_node", "#node_" + TREE_lastSelectedNode);
        } else {
            treeElem.jstree("select_node", "#projectID");
        }
        TVB_skipDisplayOverlay = false;
}
/**
 * Used for updating the tree structure.
 */
function updateTree(treeSelector, projectId, visibilityFilter) {

    if (!projectId) {
        projectId = $("#hiddenProjectId").val();
    }
    if (!visibilityFilter) {
        visibilityFilter = _getSelectedVisibilityFilter();
    }
    var firstLevel = $("#levelTree_1").val();
    var secondLevel = $("#levelTree_2").val();
    var filterValue = $("#filterInput").val();

    var url = "/project/readjsonstructure/" + projectId + "/" + visibilityFilter;

    // the dropdowns will not exists if the overlay is launched from burst or operations etc
    if (firstLevel != null && secondLevel != null){
        if (firstLevel === secondLevel) {
            displayMessage("The filters should not have the same value.", 'warningMessage');
            return;
        }
        url += "/" + firstLevel + "/" + secondLevel;
        if (filterValue.length > 2) {
            url += "/" + filterValue;
        }
    }

    $(treeSelector).jstree({
        //contextmenu: { "items" : createDefaultMenu, "select_node" : true },
        "plugins": ["themes", "json_data", "ui", "crrm"], //, "contextmenu"
        "themes": {
            "theme": "default",
            "dots": true,
            "icons": true,
            "url": deploy_context + "/static/jquery/jstree-theme/style.css"
        },
        "json_data": {
            "ajax": { url: deploy_context + url,
                success: function (d) {
                    return eval(d);
                }
            }
        }
    });
    _postInitializeTree(treeSelector);
    if (filterValue !=null && filterValue.length <= 2) {
        if (filterValue.length > 0) {
            displayMessage("You have to introduce at least tree letters in order to filter the data.", 'infoMessage');
        }
    } else {
        TREE_lastSelectedNode = undefined;
        TREE_lastSelectedNodeType = undefined;
    }
}

/**
 * Main function for specifying JSTree attributes.
 */
function _postInitializeTree(treeSelector) {
    $(treeSelector).bind("select_node.jstree", function (event, data) {

          TREE_lastSelectedNode = data.rslt.obj.attr("gid");
          if (TREE_lastSelectedNode != undefined && TREE_lastSelectedNode != null) {
              TREE_lastSelectedNodeType = TVB_NODE_DATATYPE_TYPE;
          } else {
              TREE_lastSelectedNode = undefined;
              TREE_lastSelectedNodeType = undefined;
          }

          var backPage = 'data';
          if ($("body")[0].id === "s-burst") {
                backPage = 'burst';
          }
          displayNodeDetails(TREE_lastSelectedNode, TVB_NODE_DATATYPE_TYPE, backPage);
    }).bind("loaded.jstree", function () {
        select_tree_node();
    });
}
//-----------------------------------------------------------------------
//                   TREE Section ends here
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
//                More GENERIC functions from here
//-----------------------------------------------------------------------

function createLink(dataId, projectId) {
    doAjaxCall({
        async : false,
        type: 'GET',
        url: "/project/createlink/" + dataId +"/" + projectId,
        success: function(r) {if(r) displayMessage(r,'warningMessage'); },
        error:   function(r) {if(r) displayMessage(r,'warningMessage'); }
    });
}

function removeLink(dataId, projectId, isGroup) {
    doAjaxCall({
        async : false,
        type: 'GET',
        url: "/project/removelink/" + dataId +"/" + projectId + "/" + isGroup,
        success: function(r) {if(r) displayMessage(r,'warningMessage'); },
        error:   function(r) {if(r) displayMessage(r,'warningMessage'); }
    });
}

function updateLinkableProjects(datatype_id, isGroup, entity_gid) {
    doAjaxCall({
        async : false,
        type: 'GET',
        url: "/project/get_linkable_projects/" + datatype_id + "/" + isGroup + "/" + entity_gid,
        success: function(data) {
            var linkedDiv = $("#linkable_projects_div_" + entity_gid);
            linkedDiv.empty();
            linkedDiv.append(data);
        }
    });
}

/**
 * Submits the page to the action url using a http post
 * @param params a dict with the request parameters {name:value}
 * @param action Form action string
 */
function tvbSubmitPage(action, params){
    if (TVB_pageSubmitted){
        return;
    }
    var input;
    var form = document.createElement("form");

    form.method="POST" ;
    form.action = deploy_context + action;

    for (var name in params){
        if(params.hasOwnProperty(name)){
            input = document.createElement("input");
            input.setAttribute("name", name);
            input.setAttribute("value", params[name]);
            input.setAttribute("type", "hidden");
            form.appendChild(input);
        }
    }
    document.body.appendChild(form);
    TVB_pageSubmitted = true;
    form.submit();
    document.body.removeChild(form);
}

function tvbSubmitPageAsync(action, params){
    doAjaxCall({
        url: action,
        data: params,
        success: function (data) {
            displayMessage("Operation launched.", "importantMessage");
        },
        error: function (){
            displayMessage("Operation failed to launch.", "errorMessage");
        }
    });
}

/**
 * Launch from DataType overlay an analysis or a visualize algorithm.
 */
function launchAdapter(adapter_url, param_name, param_val, back_page_link, launchAsync){
    var params = {};
    params[param_name] = param_val;
    params['fill_defaults'] = true;

    if (launchAsync){
        tvbSubmitPageAsync(adapter_url, params);
    }else {
        tvbSubmitPage(adapter_url + "?back_page=" + back_page_link, params);
    }
}

/**
 * Called when the visibility filter is changed.
 *
 * @param projectId a project id.
 */
function changedVisibilityFilter(projectId, filterElemId) {
    // Activate visibility filter
    $("#visibilityFiltersId > li[class='active']").each(function () {
        $(this).removeClass('active');
    });
    //do NOT use jquery ($("#" + filterElemId)) to select the element because its id may contain spaces
    document.getElementById(filterElemId).classList.add('active');

    TREE_lastSelectedNode = undefined;
    TREE_lastSelectedNodeType = undefined;
    updateTree('#treeStructure', projectId);
}

function _getSelectedVisibilityFilter() {
    var selectedFilter = "";
    $("#visibilityFiltersId > li[class='active']").each(function() {
        selectedFilter = this.id;
    });
    return selectedFilter;
}
