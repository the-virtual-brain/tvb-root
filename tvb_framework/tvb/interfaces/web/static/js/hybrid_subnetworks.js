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

/**
 * Groups the Connectivity regions of a Hybrid Simulation into Subnetworks.
 *
 * The server is the single source of truth: every change is sent to the Hybrid Simulator controller,
 * which validates it and answers with the resulting configuration. The board is then re-rendered from
 * that answer, so the displayed grouping can never drift away from the stored one.
 *
 * Regions are identified by their original Connectivity node index, which is never changed here.
 */

/* globals doAjaxCall, displayMessage, showQuestionOverlay */

var HYBRID_SUBNETWORKS = (function () {
    "use strict";

    const URLS = {
        add: "/burst/hybrid/add_subnetwork/",
        remove: "/burst/hybrid/remove_subnetwork/",
        rename: "/burst/hybrid/rename_subnetwork/",
        move: "/burst/hybrid/move_regions/"
    };

    const state = {
        regionLabels: [],
        subnetworks: [],
        // original Connectivity node indices of the currently selected regions
        selected: [],
        // last clicked region, used as anchor for Shift range selection
        anchor: null
    };

    let board = null;
    let summary = null;
    // the elements whose listeners are already attached, so init stays idempotent
    let boundBoard = null;
    let boundAddButton = null;

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

    function renderRegion(nodeIndex) {
        const region = createElement("li", "hybrid-region");
        region.setAttribute("draggable", "true");
        region.dataset.nodeIndex = nodeIndex;
        region.appendChild(createElement("span", "hybrid-region-index", nodeIndex));
        region.appendChild(createElement("span", "hybrid-region-label", state.regionLabels[nodeIndex]));
        region.title = state.regionLabels[nodeIndex] + " (Connectivity node " + nodeIndex + ")";
        return region;
    }

    function renderSubnetworkHeader(subnetwork, index) {
        // A plain div, never a <header>: base.css styles the bare header element as the site's top
        // navigation bar (position: fixed; top: 0; width: 100%), which would tear this title bar out
        // of its Subnetwork box and pin it to the viewport.
        const header = createElement("div", "hybrid-subnetwork-header");

        const nameInput = document.createElement("input");
        nameInput.type = "text";
        nameInput.className = "hybrid-subnetwork-name";
        nameInput.value = subnetwork.name;
        nameInput.title = "Rename this Subnetwork";
        nameInput.addEventListener("change", function () {
            renameSubnetwork(index, nameInput.value);
        });
        nameInput.addEventListener("keydown", function (event) {
            if (event.key === "Enter") {
                // do not submit the wizard form
                event.preventDefault();
                nameInput.blur();
            }
        });
        header.appendChild(nameInput);

        // Deliberately not a TVB ".action action-delete": inside the narrow Subnetwork box its sprite
        // icon (positioned at left:-12px) would overflow onto the name input.
        const removeButton = createElement("button", "hybrid-remove-subnetwork", "\u00d7");
        removeButton.type = "button";
        removeButton.title = "Remove this Subnetwork";
        removeButton.setAttribute("aria-label", "Remove this Subnetwork");
        removeButton.addEventListener("click", function () {
            confirmRemoveSubnetwork(index);
        });
        header.appendChild(removeButton);

        return header;
    }

    function renderSubnetworkActions(subnetwork, index) {
        const actions = createElement("div", "hybrid-subnetwork-actions");

        const count = subnetwork.node_indices.length;
        const countLabel = createElement("span", "hybrid-region-count" + (count === 0 ? " is-empty" : ""),
            count === 0 ? "no region" : count + (count === 1 ? " region" : " regions"));
        actions.appendChild(countLabel);

        const selectAll = createElement("button", "hybrid-subnetwork-action hybrid-select-all", "Select all");
        selectAll.type = "button";
        selectAll.disabled = count === 0;
        selectAll.title = "Select or deselect every region of this Subnetwork";
        selectAll.addEventListener("click", function () {
            toggleSubnetworkSelection(index);
        });
        actions.appendChild(selectAll);

        return actions;
    }

    function renderSubnetwork(subnetwork, index) {
        const section = createElement("section", "hybrid-subnetwork");
        section.dataset.index = index;

        section.appendChild(renderSubnetworkHeader(subnetwork, index));
        section.appendChild(renderSubnetworkActions(subnetwork, index));

        const list = createElement("ul", "hybrid-region-list");
        subnetwork.node_indices.forEach(function (nodeIndex) {
            list.appendChild(renderRegion(nodeIndex));
        });
        section.appendChild(list);

        return section;
    }

    function render() {
        if (board === null) {
            return;
        }
        board.innerHTML = "";
        state.subnetworks.forEach(function (subnetwork, index) {
            board.appendChild(renderSubnetwork(subnetwork, index));
        });
        refreshSelection();
    }

    /**
     * Only updates the decoration of the already rendered regions, so that changing the selection
     * does not rebuild the board (and does not interrupt an ongoing drag).
     */
    function refreshSelection() {
        if (board === null) {
            return;
        }
        const selected = state.selected;
        board.querySelectorAll(".hybrid-region").forEach(function (region) {
            const isSelected = selected.indexOf(parseInt(region.dataset.nodeIndex, 10)) !== -1;
            region.classList.toggle("selected", isSelected);
        });
        // The Select all toggles are derived from the selection rather than kept as their own state,
        // so picking regions by hand leaves each toggle showing the truth about its Subnetwork.
        board.querySelectorAll(".hybrid-subnetwork").forEach(function (section) {
            const toggle = section.querySelector(".hybrid-select-all");
            if (toggle === null) {
                return;
            }
            const active = isFullySelected(state.subnetworks[parseInt(section.dataset.index, 10)]);
            toggle.classList.toggle("is-active", active);
            toggle.setAttribute("aria-pressed", active ? "true" : "false");
        });
        if (summary !== null) {
            summary.textContent = selected.length === 0 ? "No region selected"
                : selected.length + (selected.length === 1 ? " region selected" : " regions selected");
        }
    }

    // ------------------------------------------------------------------ selection

    /** True when every region of a Subnetwork is currently selected. Drives its Select all toggle. */
    function isFullySelected(subnetwork) {
        if (subnetwork === undefined || subnetwork.node_indices.length === 0) {
            return false;
        }
        return subnetwork.node_indices.every(function (nodeIndex) {
            return state.selected.indexOf(nodeIndex) !== -1;
        });
    }

    /** Select every region of a Subnetwork, or deselect them all when they are already selected. */
    function toggleSubnetworkSelection(index) {
        const subnetwork = state.subnetworks[index];
        if (subnetwork === undefined || subnetwork.node_indices.length === 0) {
            return;
        }
        if (isFullySelected(subnetwork)) {
            setSelection(state.selected.filter(function (nodeIndex) {
                return subnetwork.node_indices.indexOf(nodeIndex) === -1;
            }));
        } else {
            setSelection(state.selected.concat(subnetwork.node_indices));
        }
        state.anchor = null;
    }

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

    function subnetworkOf(nodeIndex) {
        for (let i = 0; i < state.subnetworks.length; i++) {
            if (state.subnetworks[i].node_indices.indexOf(nodeIndex) !== -1) {
                return i;
            }
        }
        return -1;
    }

    function rangeBetween(fromNode, toNode) {
        const subnetworkIndex = subnetworkOf(fromNode);
        if (subnetworkIndex === -1 || subnetworkIndex !== subnetworkOf(toNode)) {
            return [toNode];
        }
        const nodes = state.subnetworks[subnetworkIndex].node_indices;
        const from = nodes.indexOf(fromNode);
        const to = nodes.indexOf(toNode);
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

    // ------------------------------------------------------------------ drag and drop

    function clearDropTargets() {
        board.querySelectorAll(".hybrid-subnetwork").forEach(function (section) {
            section.classList.remove("drop-target");
        });
    }

    function onDragStart(event) {
        const region = event.target.closest(".hybrid-region");
        if (region === null) {
            return;
        }
        const nodeIndex = parseInt(region.dataset.nodeIndex, 10);
        if (state.selected.indexOf(nodeIndex) === -1) {
            setSelection([nodeIndex]);
            state.anchor = nodeIndex;
        }
        if (event.dataTransfer) {
            event.dataTransfer.effectAllowed = "move";
            // Firefox starts a drag only when some data is set
            event.dataTransfer.setData("text/plain", JSON.stringify(state.selected));
        }
    }

    function onDragOver(event) {
        const section = event.target.closest(".hybrid-subnetwork");
        if (section === null) {
            return;
        }
        event.preventDefault();
        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = "move";
        }
        if (!section.classList.contains("drop-target")) {
            clearDropTargets();
            section.classList.add("drop-target");
        }
    }

    function onDrop(event) {
        const section = event.target.closest(".hybrid-subnetwork");
        clearDropTargets();
        if (section === null) {
            return;
        }
        event.preventDefault();
        moveSelectedTo(parseInt(section.dataset.index, 10));
    }

    // ------------------------------------------------------------------ server calls

    function onServerAnswer(response, silentWhenOk) {
        const answer = typeof response === "string" ? JSON.parse(response) : response;

        if (answer.region_labels && answer.region_labels.length) {
            state.regionLabels = answer.region_labels;
        }
        state.subnetworks = answer.subnetworks || [];

        // keep selected only the regions that are still known
        const known = [];
        state.subnetworks.forEach(function (subnetwork) {
            known.push.apply(known, subnetwork.node_indices);
        });
        state.selected = state.selected.filter(function (nodeIndex) {
            return known.indexOf(nodeIndex) !== -1;
        });

        render();

        if (answer.status !== "ok") {
            displayMessage(answer.message, "errorMessage");
        } else if (!silentWhenOk) {
            displayMessage(answer.message, "infoMessage");
        }
    }

    function callServer(url, data, silentWhenOk) {
        doAjaxCall({
            type: "POST",
            url: url,
            data: data,
            success: function (response) {
                onServerAnswer(response, silentWhenOk);
            },
            error: function () {
                displayMessage("The Subnetwork configuration could not be updated.", "errorMessage");
            }
        });
    }

    function addSubnetwork() {
        callServer(URLS.add, {});
    }

    function renameSubnetwork(index, name) {
        if (state.subnetworks[index] !== undefined && state.subnetworks[index].name === name.trim()) {
            return;
        }
        callServer(URLS.rename, {subnetwork_index: index, name: name});
    }

    function removeSubnetwork(index) {
        callServer(URLS.remove, {subnetwork_index: index});
    }

    function confirmRemoveSubnetwork(index) {
        const subnetwork = state.subnetworks[index];
        if (subnetwork === undefined) {
            return;
        }
        if (subnetwork.node_indices.length === 0 || state.subnetworks.length < 2) {
            // nothing to reassign, or the server will refuse to leave no Subnetwork behind
            removeSubnetwork(index);
            return;
        }
        const fallback = state.subnetworks[index === 0 ? 1 : 0];
        showQuestionOverlay("'" + subnetwork.name + "' still holds " + subnetwork.node_indices.length +
            " regions. They will be moved into '" + fallback.name + "'. Remove it anyway?",
            "HYBRID_SUBNETWORKS.removeSubnetwork(" + index + ")");
    }

    function moveSelectedTo(index) {
        if (state.selected.length === 0) {
            displayMessage("Select at least one Connectivity region first.", "warningMessage");
            return;
        }
        const target = state.subnetworks[index];
        if (target !== undefined && state.selected.every(function (nodeIndex) {
            return target.node_indices.indexOf(nodeIndex) !== -1;
        })) {
            // everything is already there, nothing to do
            return;
        }
        callServer(URLS.move, {subnetwork_index: index, node_indices: JSON.stringify(state.selected)}, true);
    }

    // ------------------------------------------------------------------ setup

    /**
     * The Subnetwork grouping is a dedicated step, so the cockpit's History and Results columns are
     * folded away and the configuration column takes the whole page width. The class is dropped again
     * by the wizard when any other fragment is rendered, which restores the three column layout.
     */
    const FULL_WIDTH_CLASS = "hybrid-subnetworks-step";
    const SINGLE_COLUMN_CLASS = "colscheme-1";
    // the cockpit column scheme declared by the page, remembered so it can be put back
    let cockpitColumnScheme = null;

    function setFullWidthLayout(enabled) {
        const mainDiv = document.getElementById("main");
        if (mainDiv === null) {
            return;
        }
        if (enabled) {
            if (cockpitColumnScheme === null) {
                const declared = mainDiv.className.match(/colscheme-[\w-]+/);
                cockpitColumnScheme = declared === null ? "" : declared[0];
            }
            if (cockpitColumnScheme !== "") {
                mainDiv.classList.remove(cockpitColumnScheme);
            }
            mainDiv.classList.add(SINGLE_COLUMN_CLASS, FULL_WIDTH_CLASS);
        } else {
            mainDiv.classList.remove(SINGLE_COLUMN_CLASS, FULL_WIDTH_CLASS);
            if (cockpitColumnScheme) {
                mainDiv.classList.add(cockpitColumnScheme);
            }
        }
    }

    function init(configuration) {
        board = document.getElementById("hybrid-subnetworks-board");
        summary = document.getElementById("hybrid-selection-summary");
        if (board === null) {
            return;
        }
        setFullWidthLayout(true);

        state.regionLabels = configuration.region_labels || [];
        state.subnetworks = configuration.subnetworks || [];
        state.selected = [];
        state.anchor = null;

        // Bind once per board element. Normally each render brings a brand new board, but binding
        // twice to the same one would fire every handler twice, which silently cancels out a
        // Ctrl+click toggle instead of failing loudly.
        if (boundBoard !== board) {
            board.addEventListener("click", function (event) {
                const region = event.target.closest(".hybrid-region");
                if (region !== null) {
                    onRegionClick(event, parseInt(region.dataset.nodeIndex, 10));
                }
            });
            board.addEventListener("dragstart", onDragStart);
            board.addEventListener("dragover", onDragOver);
            board.addEventListener("drop", onDrop);
            board.addEventListener("dragend", clearDropTargets);
            board.addEventListener("dragleave", function (event) {
                if (event.target === board) {
                    clearDropTargets();
                }
            });
            boundBoard = board;
        }

        const addButton = document.getElementById("hybrid-add-subnetwork");
        if (addButton !== null && boundAddButton !== addButton) {
            addButton.addEventListener("click", addSubnetwork);
            boundAddButton = addButton;
        }
        render();
    }

    return {
        init: init,
        // restores the cockpit three column layout, called when another wizard fragment is rendered
        resetLayout: function () {
            setFullWidthLayout(false);
        },
        // called from the confirmation overlay
        removeSubnetwork: removeSubnetwork,
        // exposed for tests and debugging
        _state: state
    };
})();
