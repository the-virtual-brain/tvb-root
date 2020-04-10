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

/**
 * Project Data Structure (Tree/Graph) functions are here.
 */
var GRAPH_TAB = "graphTab";
var TREE_TAB = "treeTab";

var TREE_lastSelectedNode = undefined;
var TREE_lastSelectedNodeType = undefined;
//we need this flag because we want to know when the user
// presses more times on the same graph node
var skipReload = false;

/**
 * Function selecting previously selected TAB between TREE / GRAPH, on pahe refresh.
 */
function displaySelectedTab() {

    var lastSelectedTab = $("#lastVisibleTab").val();
    if (lastSelectedTab == GRAPH_TAB) {
        showGraph();
    } else {
        showTree();
    }
}


//-----------------------------------------------------------------------
//                   TREE Section starts here
//-----------------------------------------------------------------------

function showTree() {
    $("#lastVisibleTab").val(TREE_TAB);
    $("#tabTree").show();
    $("#tabWorkflow").hide();

    $("#tree-related-li").show();

    $("#" + TREE_TAB).addClass("active");
    $("#" + GRAPH_TAB).removeClass("active");
    select_tree_node();
}


/**
 * Selects into the tree view the last node that was selected into the graph view. If
 * into the graph view was selected an operation node then into the tree view will
 * be selected the project node.
 */
function select_tree_node(treeId) {
    if ($("#lastVisibleTab").val() == TREE_TAB) {
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
        if (firstLevel == secondLevel) {
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
            "url": "/static/jquery/jstree-theme/style.css"
        },
        "json_data": {
            "ajax": { url: url,
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
          if ($("#lastVisibleTab").val() == GRAPH_TAB) {
              return;
          }
          TREE_lastSelectedNode = data.rslt.obj.attr("gid");
          if (TREE_lastSelectedNode != undefined && TREE_lastSelectedNode != null) {
              TREE_lastSelectedNodeType = TVB_NODE_DATATYPE_TYPE;
          } else {
              TREE_lastSelectedNode = undefined;
              TREE_lastSelectedNodeType = undefined;
          }
          skipReload = false;
          
          var backPage = 'data';
          if ($("body")[0].id == "s-burst") {
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
//                   GRAPH Section starts here
//-----------------------------------------------------------------------

function showGraph() {
    $("#lastVisibleTab").val(GRAPH_TAB);
    $("#tabWorkflow").show();
    $("#tabTree").hide();

    $("#tree-related-li").hide();

    $("#" + TREE_TAB).removeClass("active");
    $("#" + GRAPH_TAB).addClass("active");
    update_workflow_graph('workflowCanvasDiv', TREE_lastSelectedNode, TREE_lastSelectedNodeType);
}


function update_workflow_graph(containerDivId, nodeGid, nodeType) {
    if (nodeGid == undefined || nodeType == undefined) {
        nodeGid = "firstOperation";
        nodeType = TVB_NODE_OPERATION_TYPE;
    }
    if (nodeGid == TREE_lastSelectedNode && skipReload) {
        return;
    }
    TREE_lastSelectedNode = nodeGid;
    TREE_lastSelectedNodeType = nodeType;
    var visibilityFilter = _getSelectedVisibilityFilter();
    doAjaxCall({    async : false,
                type: 'GET',
                url: '/project/create_json/' + nodeGid + "/" + nodeType + "/" + visibilityFilter,
                success: function(data) {
                    $("#" + containerDivId).empty();
                    var json = $.parseJSON(data);
                    _draw_graph(containerDivId, json);
                }
            });

}

/**
 * Main function for specifying JIT Graph attributes.
 */
function _draw_graph(containerDivId, json) {
    // init ForceDirected
    var fd = new $jit.ForceDirected({
        //id of the visualization container
        injectInto: containerDivId,
        //Enable zooming and panning with scrolling and DnD
        Navigation: {
            enable: true,
            //Enable panning events only if we're dragging the empty canvas (and not a node).
            panning: 'avoid nodes',
            zooming: 10
        },
        // Change node and edge styles such as color and width.
        // These properties are also set per node with dollar prefixed data-properties in the JSON structure.
        Node: {
            overridable: true
        },
        Edge: {
            overridable: true,
            type: 'arrow',
            color: '#23A4FF',
            lineWidth: 0.4
        },
        Tips: {
            enable: true,
            type: 'Native',
            onShow: function(tip, node) {
                if (node.id != "fakeRootNode") {
                    tip.innerHTML = "<div class=\"tip-title\">" + node.name + "</div></br>" +
                                    "<div class=\"tip-text\">" +
                                            "<b>Id:</b> " + node.data.node_entity_id + "</br>" +
                                            "<b>Type:</b> " + node.data.node_type + "</br>" +
                                            "<b>Display Name:</b> " + node.data.node_subtitle + "</br>" +
                                    "</div>";
                    }
            }
        },
        // Add node events
        Events: {
            enable: true,
            type: 'Native',
            //Change cursor style when hovering a node
            onMouseEnter: function(node, eventInfo, e) {
                fd.canvas.getElement().style.cursor = 'move';
            },
            onMouseLeave: function(node, eventInfo, e) {
                fd.canvas.getElement().style.cursor = '';
            },
            //Update node positions when dragged
            onDragMove: function(node, eventInfo, e) {
                var pos = eventInfo.getPos();
                node.pos.setc(pos.x, pos.y);
                fd.plot();
            },
            //Implement the same handler for touchscreens
            onTouchMove: function(node, eventInfo, e) {
                //stop default touchmove event
                $jit.util.event.stop(e);
                this.onDragMove(node, eventInfo, e);
            }
        },
        // This method is only triggered
        // on label creation and only for DOM labels (not native canvas ones).
        onCreateLabel: function(domElement, node) {
            var nameContainer = document.createElement('span');
            var style = nameContainer.style;

            //1 character is drawn on 3 points
            var nodeName = node.name;
            if (node.data.$dim < 30) {
                nodeName = nodeName.substr(0, 3) + "...";
            } else {
                if (nodeName.length * 3 > node.data.$dim) {
                    nodeName = nodeName.substr(0, (node.data.$dim / 3) - 3);
                    nodeName += "...";
                }
            }

            if (node.id == "fakeRootNode") {
                node.setData('alpha', 0);
                node.eachAdjacency(function(adj) {
                    adj.setData('alpha', 0);
                });
                return;
            }
            nameContainer.className = 'name';
            nameContainer.innerHTML = nodeName;
            domElement.appendChild(nameContainer);

            style.fontSize = "1.0em";
            style.color = "#ddd";
            if (node.id == TREE_lastSelectedNode) {
                node.setData('color', "#ff0000");
            }

            nameContainer.onclick = function() {
                update_workflow_graph(containerDivId, node.id, node.data.node_type);
            };
        },
        // Change node styles when DOM labels are placed or moved.
        onPlaceLabel: function(domElement, node) {
            var style = domElement.style;
            var left = parseInt(style.left);
            var top = parseInt(style.top);
            var w = domElement.offsetWidth;
            style.left = (left - w / 2) + 'px';
            style.top = (top - 5) + 'px';
            style.display = '';
        }
    });
    // load JSON data.
    fd.loadJSON(json);
    // compute positions incrementally and animate.
    fd.computeIncremental({
        property: 'end',
        onComplete: function() {
            fd.animate({
                modes: ['linear'],
                transition: $jit.Trans.Elastic.easeOut,
                duration: 1500
            });
        }
    });
}

//-----------------------------------------------------------------------
//                   GRAPH Section ends here
//-----------------------------------------------------------------------


//-----------------------------------------------------------------------
//                More GENERIC functions from here
//-----------------------------------------------------------------------

function createLink(dataId, projectId, isGroup) {
    doAjaxCall({
        async : false,
        type: 'GET',
        url: "/project/createlink/" + dataId +"/" + projectId + "/" + isGroup,
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
    form.action = action;

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
    update_workflow_graph('workflowCanvasDiv', TREE_lastSelectedNode, TREE_lastSelectedNodeType);
    updateTree('#treeStructure', projectId);
}

function _getSelectedVisibilityFilter() {
    var selectedFilter = "";
    $("#visibilityFiltersId > li[class='active']").each(function() {
        selectedFilter = this.id;
    });
    return selectedFilter;
}


