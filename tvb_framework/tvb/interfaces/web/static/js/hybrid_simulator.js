(function () {
    function buildGroups() {
        const root = document.getElementById('hybrid-groups');
        const hidden = document.getElementById('hybrid-subnetworks');
        if (!root || !hidden) return;
        let groups = window.HYBRID_SUBNETWORKS || [];
        if (!groups.length && (window.HYBRID_LABELS || []).length) {
            groups = [{name: 'Subnetwork 1', nodes: []}, {name: 'Subnetwork 2', nodes: []}];
        }
        function collectGroups() {
            groups = Array.from(root.querySelectorAll('fieldset')).map(function (section) {
                return {name: section.querySelector('.hybrid-group-name').value, nodes: Array.from(section.querySelectorAll('input[type=checkbox]:checked')).map(function (box) { return Number(box.dataset.node); })};
            });
        }
        function render() {
            root.innerHTML = '';
            groups.forEach(function (group, groupIndex) {
                const section = document.createElement('fieldset');
                section.innerHTML = '<legend><input class="hybrid-group-name" value="' + group.name + '"> <button type="button" class="hybrid-select-all" data-group="' + groupIndex + '">Select all</button> <button type="button" class="hybrid-remove-group" data-group="' + groupIndex + '">Remove</button></legend>';
                (window.HYBRID_LABELS || []).forEach(function (label, nodeIndex) {
                    const id = 'hybrid-node-' + groupIndex + '-' + nodeIndex;
                    const checked = group.nodes.indexOf(nodeIndex) >= 0 ? ' checked' : '';
                    section.insertAdjacentHTML('beforeend', '<label class="hybrid-node"><input type="checkbox" id="' + id + '" data-node="' + nodeIndex + '"' + checked + '>' + label + '</label>');
                });
                root.appendChild(section);
            });
            root.querySelectorAll('input[type=checkbox]').forEach(function (box) {
                box.addEventListener('change', function () {
                    if (box.checked) root.querySelectorAll('input[data-node="' + box.dataset.node + '"]').forEach(function (other) { if (other !== box) other.checked = false; });
                    const checked = new Set(Array.from(root.querySelectorAll('input[type=checkbox]:checked')).map(function (item) { return Number(item.dataset.node); }));
                    document.getElementById('hybrid-unassigned').textContent = (window.HYBRID_LABELS || []).filter(function (_, index) { return !checked.has(index); }).join(', ') || 'None';
                });
            });
            root.querySelectorAll('.hybrid-remove-group').forEach(function (button) {
                button.addEventListener('click', function () { collectGroups(); groups.splice(Number(button.dataset.group), 1); render(); });
            });
            root.querySelectorAll('.hybrid-select-all').forEach(function (button) {
                button.addEventListener('click', function () {
                    root.querySelectorAll('input[type=checkbox]').forEach(function (box) {
                        box.checked = box.id.indexOf('hybrid-node-' + button.dataset.group + '-') === 0;
                    });
                    collectGroups();
                    render();
                });
            });
            const assigned = new Set(groups.reduce(function (nodes, group) { return nodes.concat(group.nodes); }, []));
            document.getElementById('hybrid-unassigned').textContent = (window.HYBRID_LABELS || []).filter(function (_, index) { return !assigned.has(index); }).join(', ') || 'None';
        }
        document.getElementById('hybrid-add-subnetwork').addEventListener('click', function () { collectGroups(); groups.push({name: 'Subnetwork ' + (groups.length + 1), nodes: []}); render(); });
        document.getElementById('hybrid-connectivity-form').addEventListener('submit', function () {
            collectGroups();
            hidden.value = JSON.stringify(groups);
        });
        render();
    }
    function bindLaunch() {
        const button = document.getElementById('hybrid-launch');
        if (!button) return;
        button.addEventListener('click', function () {
            button.disabled = true;
            fetch(button.dataset.launchUrl, {method: 'POST'}).then(function (response) {
                if (!response.ok) throw new Error('The server could not launch the simulation.');
                return response.json();
            }).then(function (result) {
                if (result.error) { button.disabled = false; displayMessage(result.error, 'errorMessage'); return; }
                window.location = button.dataset.burstUrl + '?burst_id=' + result.id;
            }).catch(function (error) {
                button.disabled = false;
                displayMessage(error.message, 'errorMessage');
            });
        });
    }
    function buildMatrix(textareaId, rootId) {
        const textarea = document.getElementById(textareaId);
        const root = document.getElementById(rootId);
        if (!textarea || !root) return;
        const values = JSON.parse(textarea.value);
        const table = document.createElement('table');
        const header = document.createElement('tr');
        header.appendChild(document.createElement('th'));
        (window.HYBRID_SOURCE_LABELS || []).forEach(function (label) { const th = document.createElement('th'); th.textContent = label; header.appendChild(th); });
        table.appendChild(header);
        values.forEach(function (row, rowIndex) {
            const tr = document.createElement('tr');
            const th = document.createElement('th'); th.textContent = (window.HYBRID_TARGET_LABELS || [])[rowIndex]; tr.appendChild(th);
            row.forEach(function (value, columnIndex) {
                const td = document.createElement('td');
                const input = document.createElement('input'); input.type = 'number'; input.step = 'any'; input.value = value; input.dataset.row = rowIndex; input.dataset.column = columnIndex;
                td.appendChild(input); tr.appendChild(td);
            });
            table.appendChild(tr);
        });
        root.appendChild(table);
        return function () {
            root.querySelectorAll('input').forEach(function (input) { values[Number(input.dataset.row)][Number(input.dataset.column)] = Number(input.value); });
            textarea.value = JSON.stringify(values);
        };
    }
    function buildProjectionEditor() {
        const form = document.getElementById('hybrid-projection-form');
        if (!form) return;
        const saveWeights = buildMatrix('weights', 'hybrid-weights-grid');
        const saveTracts = buildMatrix('tract_lengths', 'hybrid-tracts-grid');
        form.addEventListener('submit', function () { saveWeights(); saveTracts(); });
    }
    function initialize() {
        buildGroups();
        buildProjectionEditor();
        bindLaunch();
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }
}());
