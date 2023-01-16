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

/*globals displayMessage,GFUNC_isNodeAddedToInterestArea */

/*
 * Variables and function for selecting and applying operations are defined below
 */

/*
 * The following constants define a operation done between nodes:
 * 1 - that are both in the interest area
 * 2 - that have the source in interest area and destination outside
 * 3 - that have the source outside interest area and destination inside
 */
SEL_INTER_OPERATION = 1;
SEL_EXIT_OPERATION = 2;
SEL_ENTER_OPERATION = 3;
SEL_OUTER_OPERATION = 4;

/*
 * Just dummy functions used for testing new 'selection operation' To be removed after.
 */

function divideSelectionBy(input, quantity) {
    return input / quantity;
}

function multiplySelectionWith(input, quantity) {
    return input * quantity;
}

function addToSelection(input, quantity) {
    return input + quantity;
}

function decreaseSelection(input, quantity) {
    return input - quantity;
}

function setSelection(input, quantity) {
    return quantity;
}

OP_DICTIONARY ={ 1 : { 'name': 'Set(n)', 'operation': setSelection },
                 2 : { 'name': 'Add(n)', 'operation': addToSelection },
                 3 : { 'name': 'Decrease(n)', 'operation': decreaseSelection },
                 4 : { 'name': 'Multiply(n)', 'operation': multiplySelectionWith },
                 5 : { 'name': 'Divide(n)', 'operation': divideSelectionBy } }

EDGES_TYPES = { 1 : 'In --> In',
				2 : 'In --> Out',
                3 : 'Out --> In',
				4 : 'Out --> Out' };


function SEL_createOperationsTable() {
    var operationsSelect = document.getElementById("con-op-operation");
    var index, option;
    for (index in OP_DICTIONARY) {
        option = new Option(OP_DICTIONARY[index]['name'], index);
        operationsSelect.options[operationsSelect.options.length] = option;
    }

    var edgesTypeSelect = document.getElementById("con-op-edges-type");
    for (index in EDGES_TYPES) {
        option = new Option(EDGES_TYPES[index], index);
        edgesTypeSelect.options[edgesTypeSelect.options.length] = option;
    }
}

function getOperationArguments() {
    //TODO: if new functions will be needed with multiple arguments this should be edited
    return parseFloat(document.getElementById('con-op-arguments').value);
}

function doGroupOperation() {
    //Selected operation
    var operationsSelect = document.getElementById('con-op-operation');
    var selectedOp = parseInt(operationsSelect.options[operationsSelect.selectedIndex].value);
    selectedOp = OP_DICTIONARY[selectedOp]['operation'];
    //Selected edges type
    var edgesSelect = document.getElementById('con-op-edges-type');
    var selectedEdgeType = parseInt(edgesSelect.options[edgesSelect.selectedIndex].value);
    //Arguments and results label
    var quantity = getOperationArguments();

    document.getElementById('con-op-arguments').value = '';
    if (isNaN(quantity)) {
        displayMessage("Operation failed. Be sure you provided the correct arguments.", "errorMessage");
        return false;
    }

    // operate on currently visible matrix
    var values = GVAR_interestAreaVariables[GVAR_selectedAreaType].values;

    try {
        for (var i=0; i < values.length; i++) {
            for (var j=0; j < values[i].length; j++) {
                switch(selectedEdgeType) {
                    case 1:
                        if (GFUNC_isNodeAddedToInterestArea(i) && GFUNC_isNodeAddedToInterestArea(j)) {
                            values[i][j] = selectedOp(values[i][j], quantity);
                        }
                        break;
                    case 2:
                        if (GFUNC_isNodeAddedToInterestArea(i) && !GFUNC_isNodeAddedToInterestArea(j)) {
                            values[i][j] = selectedOp(values[i][j], quantity);
                        }
                        break;
                    case 3:
                        if (!GFUNC_isNodeAddedToInterestArea(i) && GFUNC_isNodeAddedToInterestArea(j)) {
                            values[i][j] = selectedOp(values[i][j], quantity);
                        }
                        break;
                    case 4:
                        if (!GFUNC_isNodeAddedToInterestArea(i) && !GFUNC_isNodeAddedToInterestArea(j)) {
                            values[i][j] = selectedOp(values[i][j], quantity);
                        }
                        break;
                }
            }
        }
        GFUNC_recomputeMinMaxW();
        MATRIX_colorTable();
        displayMessage("Operation finished successfully.", "infoMessage");
    } catch(err) {
        displayMessage("Operation failed. Be sure you provided the correct arguments.", "errorMessage");
    }
}


