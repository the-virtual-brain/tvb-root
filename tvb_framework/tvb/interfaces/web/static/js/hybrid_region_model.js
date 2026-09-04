/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need to download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
 *
 * This program is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with this
 * program.  If not, see <http://www.gnu.org/licenses/>.
 **/

/* globals doAjaxCall, displayMessage */

/**
 * Places saved Dynamics on the regions of the Subnetwork being configured: the right hand section of
 * the classic Set up region Model page, without its 3D view.
 *
 * The classic component (TVBUI.RegionAssociatorView) is not reused because it drives that 3D view
 * through globals - GVAR_interestAreaNodeIndexes, CONN_pickedIndex, GFUNC_toggleNodeInInterestArea and
 * a #GLcanvas click handler - none of which exist on this page. The region list uses the same
 * click / Ctrl / Shift interaction as the Subnetwork grouping board instead.
 *
 * The server owns the state, as everywhere else in this wizard: every placement round-trips and the
 * answer carries the whole list back, which the browser redraws.
 */
var HYBRID_REGION_MODEL = (function () {

    const URLS = {
        apply: "/burst/hybrid/apply_region_model/",
        dynamicDetail: "/burst/dynamic/dynamic_detail/"
    };

    const state = {
        // one entry per region of this Subnetwork: index, label and the configuration placed on it
        rows: [],
        // original Connectivity node indices of the currently selected regions
        selected: [],
        // last clicked region, used as anchor for Shift range selection
        anchor: null,
        unassigned: 0
    };

    let list = null;
    let statusLabel = null;
    let selectionLabel = null;
    // the elements whose listeners are already attached, so init stays idempotent
    let boundList = null;
    let boundApply = null;
    let boundValues = null;

    // ------------------------------------------------------------------ rendering

    function createElement(tag, className, text) {
        const element = document.createElement(tag);
        if (className) {
            element.className = className;
        }
        if (text !== undefined && text !== null) {
            element.textContent = text;
        }
        return element;
    }

    function renderRow(row) {
        const item = createElement("li", "hybrid-region");
        item.dataset.nodeIndex = row.index;
        item.appendChild(createElement("span", "hybrid-region-index", row.index));
        item.appendChild(createElement("span", "hybrid-region-label", row.label));
        item.appendChild(createElement("span",
            row.dynamic_name ? "hybrid-region-dynamic" : "hybrid-region-dynamic is-empty",
            row.dynamic_name || "not configured"));
        return item;
    }

    function render() {
        if (list === null) {
            return;
        }
        list.innerHTML = "";
        state.rows.forEach(function (row) {
            list.appendChild(renderRow(row));
        });
        refreshSelection();
        refreshStatus();
    }

    function refreshStatus() {
        if (statusLabel === null) {
            return;
        }
        if (state.unassigned === 0) {
            statusLabel.textContent = "Every region is configured";
            statusLabel.classList.remove("is-modified");
        } else {
            statusLabel.textContent = state.unassigned + " region" +
                (state.unassigned === 1 ? "" : "s") + " without a configuration";
            statusLabel.classList.add("is-modified");
        }
    }

    function refreshSelection() {
        if (list !== null) {
            list.querySelectorAll(".hybrid-region").forEach(function (item) {
                const nodeIndex = parseInt(item.dataset.nodeIndex, 10);
                item.classList.toggle("selected", state.selected.indexOf(nodeIndex) !== -1);
            });
        }
        if (selectionLabel !== null) {
            selectionLabel.textContent = state.selected.length === 0
                ? "No region selected"
                : state.selected.length + " region" + (state.selected.length === 1 ? "" : "s") + " selected";
        }
    }

    // ------------------------------------------------------------------ selection

    function setSelection(nodeIndices) {
        const unique = [];
        nodeIndices.forEach(function (nodeIndex) {
            if (unique.indexOf(nodeIndex) === -1) {
                unique.push(nodeIndex);
            }
        });
        state.selected = unique;
        refreshSelection();
    }

    function orderedIndices() {
        return state.rows.map(function (row) {
            return row.index;
        });
    }

    function rangeBetween(fromNode, toNode) {
        const nodes = orderedIndices();
        const from = nodes.indexOf(fromNode);
        const to = nodes.indexOf(toNode);
        if (from === -1 || to === -1) {
            return [toNode];
        }
        return nodes.slice(Math.min(from, to), Math.max(from, to) + 1);
    }

    function onRegionClick(event, nodeIndex) {
        if (event.shiftKey && state.anchor !== null) {
            setSelection(state.selected.concat(rangeBetween(state.anchor, nodeIndex)));
            return;
        }
        if (event.ctrlKey || event.metaKey) {
            const position = state.selected.indexOf(nodeIndex);
            if (position === -1) {
                setSelection(state.selected.concat([nodeIndex]));
            } else {
                const remaining = state.selected.slice();
                remaining.splice(position, 1);
                setSelection(remaining);
            }
            state.anchor = nodeIndex;
            return;
        }
        setSelection([nodeIndex]);
        state.anchor = nodeIndex;
    }

    // ------------------------------------------------------------------ server

    function onServerAnswer(response) {
        let answer;
        try {
            answer = typeof response === "string" ? JSON.parse(response) : response;
        } catch (error) {
            displayMessage("The Model configuration could not be placed.", "errorMessage");
            return;
        }

        state.rows = answer.rows || [];
        state.unassigned = answer.unassigned || 0;
        // what was just configured stays selected, so a wrong choice can be corrected straight away
        setSelection(state.selected.filter(function (nodeIndex) {
            return orderedIndices().indexOf(nodeIndex) !== -1;
        }));
        render();

        if (answer.message) {
            displayMessage(answer.message, answer.status === "error" ? "errorMessage" : "infoMessage");
        }
    }

    function applyToSelection() {
        const selector = document.getElementById("hybrid-region-dynamic");
        if (selector === null) {
            return;
        }
        doAjaxCall({
            type: "POST",
            url: URLS.apply,
            data: {dynamic_id: selector.value, node_indices: JSON.stringify(state.selected)},
            success: onServerAnswer,
            error: function () {
                displayMessage("The Model configuration could not be placed.", "errorMessage");
            }
        });
    }

    function toggleValues() {
        const pane = document.getElementById("hybrid-region-values-pane");
        const selector = document.getElementById("hybrid-region-dynamic");
        if (pane === null || selector === null) {
            return;
        }
        if (pane.style.display !== "none") {
            pane.style.display = "none";
            return;
        }
        doAjaxCall({
            type: "GET",
            url: URLS.dynamicDetail + selector.value,
            success: function (fragment) {
                pane.innerHTML = fragment;
                pane.style.display = "";
            },
            error: function () {
                displayMessage("The values of this model configuration could not be loaded.", "errorMessage");
            }
        });
    }

    // ------------------------------------------------------------------ wiring

    function init(configuration) {
        list = document.getElementById("hybrid-region-model-list");
        statusLabel = document.getElementById("hybrid-region-status");
        selectionLabel = document.getElementById("hybrid-region-selection");

        state.rows = configuration.rows || [];
        state.unassigned = configuration.unassigned || 0;
        state.selected = [];
        state.anchor = null;

        if (list !== null && boundList !== list) {
            list.addEventListener("click", function (event) {
                const item = event.target.closest(".hybrid-region");
                if (item === null) {
                    return;
                }
                onRegionClick(event, parseInt(item.dataset.nodeIndex, 10));
            });
            boundList = list;
        }

        const applyButton = document.getElementById("hybrid-region-apply");
        if (applyButton !== null && boundApply !== applyButton) {
            applyButton.addEventListener("click", applyToSelection);
            boundApply = applyButton;
        }

        const valuesButton = document.getElementById("hybrid-region-values");
        if (valuesButton !== null && boundValues !== valuesButton) {
            valuesButton.addEventListener("click", toggleValues);
            boundValues = valuesButton;
        }

        render();
    }

    return {init: init};
})();
