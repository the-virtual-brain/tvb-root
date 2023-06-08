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
 * This file has been produced by Devin Baily, as part of GSOC 2016 program.
 *
 * this file is mean't to house the API skeleton for the new PSE data-structure. Ideally there needs to be several central features however they end up being coded.
 * --support for the addition of new results into the data-structure, and eventually onto the canvas
 * --support for iterative searching over each result and comparing its metrics to some user filter criteria
 * --support for iterative comparison of results and their neighbors within a certain distance, and determining whether the difference between the metrics of the compared results is above or below a user input criteria
 * ----this may include a method of grabbing all of the nearby results within some geometric area (circle or square) based on the step values that have been used to generate the different sets of results in the datagroup.
 *
 * Created by dev on 7/14/16.
 */
// todo make sure to ask lia about how to combine the information that is stored in PSE_nodeInfo, because that could present some issues when the  data is on the page and there isn't an actual result to access upon clicks or hovers
// todo decide whether i should make some sort of init alternate function that takes all of the information from the working state of the program before the data starts being merged or anything.


//this function will take in the original 1 dimensional array (seriesArray likely) and generate a matrix (sparse or condensed) that will be accessed upon plotting, filtering, and generating overlays.
//I'm just going to try a COO coordinate list of tuples, but then why would I change it from the current object array?
// do I want the data to be sorted at all? Maybe either by row or column parameter value?
function createStructure(data, xlabels, ylabels) {
    retCSR = {IA: [0], A: data, JA: []};
    for (var i in ylabels) { //this will populate the IA array with as many entries as the # of rows + 1
        var rowCount = 0;
        data.map(function (ob) {
            if (ob.coords.y == ylabels[i]) {
                return ++rowCount
            }
        });
        retCSR.IA.push(retCSR.IA[i] + rowCount); //this should incrementally give us the number of nonzero results in the row, adding eventuall to a final value which is the total nonzero entries
    }
    for (var ob of data) {
        retCSR.JA.push(xlabels.indexOf(ob.coords.x)); // I believe it is very important not to sort this or else the map of what ob goes with which column will be lost
    }
    return retCSR
}

// I think I should write  a function that will remove the duplicate instances of result occurence from the create structure step.
function removeDuplicateRes(data) {
    var obCounter = {},
        discardOb,
        discardInd;
    for (var ob of data) {
        discardOb = '' + ob.coords.x + '' + ob.coords.y;
        if (obCounter.hasOwnProperty(discardOb)) {
            discardInd = data.indexOf(ob);
            data.splice(discardInd, 1)
        } else {
            obCounter[discardOb] = undefined
        }
    }
    return data
}

//this function recreates a row from the sparse matrix and leaves undefined objects where there isn't a result. The tricky part here is to remember that the ithrow starts counting at 0
//this function is an inclusive slice function for the compressed sparse matrix structure
function reconstructMatrixBySection(structure, startRow, endRow, startCol, endCol) { // it's also worth bearing in mind that the 0th row is the bottom, lowest y row
    var retArr = Array(), //I shouldn't include any empty results just because it will make my life harder later on.
        startInd = structure.IA[startRow],
        endInd = structure.IA[endRow + 1],//normally there would need to be a -1 at the end here, but since slice isn't inclusive i took it away
        nzEles = structure.A.slice(startInd, endInd),
        colPos = structure.JA.slice(startInd, endInd);
    for (var i in colPos) {
        if (colPos[i] > endCol || colPos[i] < startCol) { // todo does this need to be inclusive or what?
            nzEles[i] = undefined//if the result belongs to a column that we aren't interested in then it shall be removed from the selection.
            // i think there could be a way to simply change the actual structure, and do away with the filtering below.
        }
    }
    //please note that retArr isn't the same as nzEles because they are different lengths for one, and retArr is sorted with duplicate dataentries (same parameter coords) paired down.
    //todo figure out whether the fact that the results at the same point get overwritten is going to cause problems later on.
    //well one thing is for sure, it won't be considered to be a closer result given that the difference between the distance for duplicate dots isn't >0 so it wont pass test below.
    //now i will filter out the undefineds

    retArr = nzEles.filter(function (ele) {
        if (ele != 'undefined') return ele
    }); //currently this only contains items in the row or column that we are interested in, pretty great huh?
    return retArr
}


//this function will be used to create arrays of labels separated into x and y based on a collection of data that is passed in.
function constructLabels(data) {
    var xArr = [],
        yArr = [];
    for (var ob of data) {
        var obX = ob.coords.x,
            obY = ob.coords.y;
        if (xArr.indexOf(ob.coords.x) == -1) {
            xArr.push(obX)
        }
        if (yArr.indexOf(ob.coords.y) == -1) {
            yArr.push(obY)
        }
    }
    xArr.sort();
    yArr.sort();
    return [xArr, yArr]
}

//this function will return the number of significant figures within a float, and can be used to help make sure that the correct numbers are generated through the stepDetector
function sigFigNum(flt) {
    var fltStr = flt.toExponential(),
        sigStr = fltStr.replace(/e.*$/, "");
    return sigStr.length
}

//this is just a basic updater to keep track of the step values that are present in the canvas for the separate parameters
// i don't suppose that I need to make sure that the arrays have atleast 1 entry?
// problem here is that this function only workes on arrays that have not been sorted by value, so that the steps can be next to each other.
function updateKnownSteps(stepOb, xArr, yArr) { // todo either deprecate, or switch to storing from computed explorations.
    var xStep = [],
        yStep = [];
    for (var [arr, step] of [[xArr, xStep], [yArr, yStep]]) // should suffice to make the loops general to the x&y
        for (var i = 0; i < arr.length - 2; i++) {
            var left = arr[i],
                right = arr[i + 1],
                sigLeft = sigFigNum(left),
                sigRight = sigFigNum(right),
                dif = +(right - left).toFixed(sigLeft < sigRight ? sigRight : sigLeft); // this will help to minimize returning the wrong step value ideally.
            if (Step.indexOf(dif) == -1) {
                Step.push(dif)
            }

        }

    stepOb.x = xStep;
    stepOb.y = yStep;

}

function sortResults(data) {
    data.sort(function (obA, obB) { // for CSR I think maybe I need to be sorting on the y, and it needs to sort such that the order is increasing like labels
        if (obA.coords.y < obB.coords.y) return -1;
        if (obA.coords.y > obB.coords.y) return 1;
        if (obA.coords.x < obB.coords.x) return -1;
        if (obA.coords.x > obB.coords.x) return 1
    });

    return data
}

//very important to be able to add new results into the current structure whatever winds up being selected
//this could be a good place to update a list that is storing the step values that have been used so far. Probably the call to update the coordinate arrays
// i suppose this is a lot like the addition property of traditional matrices, it just needs to be tailored to work with matrices that are different sizes (controversial).
function mergeResults(newData, data) {
    //what I did in the actual html script tag was to simply concat the two arrays into one large one, and then sort on the y parameter, will it continue to be this simple?
    dataAll = newData.concat(data)//must make into Array before I can actually perform the sort below that is necessary
    removeDuplicateRes(dataAll);
    return sortResults(dataAll);


}

//this function is a parallel to the mergeResults because it is meant to merge the information that comes in the form of nodeInfo, should just be a merging two objects deal.
function mergeNodeInfo(oldInfo, newInfo) {
    for (var xCoordAttr in newInfo) {
        for (var yCoordAttr in newInfo[xCoordAttr])
            if (oldInfo.hasOwnProperty(xCoordAttr)) {
                oldInfo[xCoordAttr][yCoordAttr] = newInfo[xCoordAttr][yCoordAttr] //this way we overwrite any cases of the same attribute combo
            } else {
                oldInfo[xCoordAttr] = {}
                oldInfo[xCoordAttr][yCoordAttr] = newInfo[xCoordAttr][yCoordAttr]
            }
    }
    return oldInfo
}


//this function will calculate the difference between each result and its neighbors upon a selected metric. This difference will be compared to a user input value, and will dictate whether there are colored contour lines added inbetween results to help visualize the change in results based on parameter change.
//don't forget that this will need to have some sort of threshold for the number of results that are coming back from the geometric selection before proceeding
//      if there happens to be too many decrease the distance value to the next smallest step value (x&y step arrays) that exists from the calculations.
//
function compareToNeighbors(structure, stepOb, xArr, yArr, srchCritOb, nodeInfo) {

    //this function returns a result that has been decided to be the closest neighbor to the currently examined dot for each parameter direction, and also according to boundary rules
    function chooseClosestNeighbor(selection, currentRes) {
        //dir is either going to be right or up in each case
        var difArr = selection.map(function (ob) {
            var xDst = ob.coords.x - currentRes.coords.x,
                yDst = (ob.coords.y - currentRes.coords.y); //must scale y diff to favor the cases in the right"er direction
            if (xDst >= 0 && yDst >= 0) {
                return Math.sqrt(xDst * xDst + yDst * yDst)
            } //why doesn't js just have a syntax for power, i don't feel like invoking math?
            return undefined
        }); // this should allow us to only get a positive float array to minimize with indexing intact for ob retrieval,
        return selection.splice(difArr.indexOf(d3.min(difArr)), 1)[0]; // return the actual object not a single element array with it inside
    }

    //this function is meant to allow for the user specified search criteria to actually be compared to the differences in metrics between current results and their neighbors.
    //the nodeInfo must be included so that we can compare with color_weight values
    function checkAgainstCriteria(srchCritOb, currentRes, neighbor, nodeInfo, structure) {

        if (srchCritOb.type == 'Size') {
            var min_size = +d3.select("#minShapeLabel").node().innerHTML,
                max_size = +d3.select("#maxShapeLabel").node().innerHTML,
                sizeScale = d3.scale.linear()
                    .range([min_size, max_size]) // these plus signs convert the string to number
                    .domain(d3.extent(structure.A, function (d) {
                        return +d.points.radius
                    })),
                diff = Math.abs(sizeScale(neighbor.points.radius) - sizeScale(currentRes.points.radius)); //absolute value, because we want to draw lines for jumps in either direction: values increasing or decreasing
            return srchCritOb.not ? diff < srchCritOb.value : diff > srchCritOb.value // this line means, if the user selected inverse search value, then we want lines drawn everywhere the difference is below the value, else only draw lines where the dif is bigger than the value
        }
        else if (srchCritOb.type == 'Color') {
            var currentColorWeight = nodeInfo[currentRes.coords.x][currentRes.coords.y].color_weight,
                neighborColorWeight = nodeInfo[neighbor.coords.x][neighbor.coords.y].color_weight,
                diffColorWeight = Math.abs(neighborColorWeight - currentColorWeight);
            return srchCritOb.not ? diffColorWeight < srchCritOb.value : diffColorWeight > srchCritOb.value
        }
    }

    var obCompTracker = [];


    for (var ob of structure.A) { //do I need to sort this by x coord?
        // create some sort of control variable for the step val index
        var currentRowInd = yArr.indexOf(ob.coords.y),
            neighbor = {},
            currentColInd = xArr.indexOf(ob.coords.x),
            topRowInd = yArr.length - 1,
            farRightColInd = xArr.length - 1,
            yStepInd = 0, //these will be used to control amount of results that get passed to the chooseClosestNeighbor
            xStepInd = 0,
            counter = 0,
            closest = [],
            switchExprssn = (currentRowInd == topRowInd) + "," + (currentColInd == farRightColInd);
        switch (switchExprssn) { //each of these will help us to determine what kind of parameters we need to include in the calls below

            // are the breaks going to exit the switch or the while?

            case ('false,false'): //all non top row or right column results
                // get the selection
                var rowBoundary = yArr.indexOf(+(ob.coords.y + stepOb.y[yStepInd]).toFixed(2)), // stepOb is arranged from large to small
                    colBoundary = xArr.indexOf(+(ob.coords.x + stepOb.x[xStepInd]).toFixed(2)),
                    selectedResults = reconstructMatrixBySection(structure, currentRowInd, rowBoundary, currentColInd, colBoundary);
                selectedResults.splice(selectedResults.indexOf(ob), 1);
                while (selectedResults.length > 10 || selectedResults.length == 0 && counter < 30) { //inner while loop is to allow the step values arr to help adjust the amount of results we get back
                    ++counter;
                    if (rowBoundary == -1) ++yStepInd;
                    if (colBoundary == -1) ++xStepInd;
                    if (selectedResults.length > 10) {
                        if (colBoundary > rowBoundary) {
                            ++xStepInd
                        } else {
                            ++yStepInd
                        }
                    }
                    if (!stepOb.x[xStepInd] || !stepOb.y[yStepInd]) break;
                    var rowBoundary = yArr.indexOf(+(ob.coords.y + stepOb.y[yStepInd]).toFixed(2)),
                        colBoundary = xArr.indexOf(+(ob.coords.x + stepOb.x[xStepInd]).toFixed(2)),
                        selectedResults = reconstructMatrixBySection(structure, currentRowInd, rowBoundary, currentColInd, colBoundary);
                    selectedResults.splice(selectedResults.indexOf(ob), 1)
                }

                for (var i = 0; i < 3; i++) {
                    if (selectedResults.length > 0) {
                        neighbor = chooseClosestNeighbor(selectedResults, ob);


                        if (checkAgainstCriteria(srchCritOb, ob, neighbor, nodeInfo, structure)) {
                            closest.push(neighbor)
                        }
                    }
                }
                ;
                // next it will be good to create some sort of actual metric comparison function to call below
                //todo maake metric comparison function
                //todo email lia asking how to get info for results metrics that arent currently available in color or rad.
                obCompTracker.push({
                    'focalPoint': ob, 'neighbors': closest.map(function (ele) {
                        return ele.coords.x + ' ' + ele.coords.y
                    })
                })
                break;
            case ('true,false'): //top row
                var colBoundary = xArr.indexOf(+(ob.coords.x + stepOb.x[xStepInd]).toFixed(2)),
                    selectedResults = reconstructMatrixBySection(structure, currentRowInd, currentRowInd, currentColInd, colBoundary); // return only items from the row
                selectedResults.splice(selectedResults.indexOf(ob), 1);

                while (selectedResults.length > 10 || selectedResults.length == 0 && counter < 30) { //inner while loop is to allow the step values arr to help adjust the amount of results we get back
                    ++counter;
                    if (colBoundary == -1) ++xStepInd;
                    if (selectedResults.length > 10) {
                        ++xStepInd
                    }
                    if (!stepOb.x[xStepInd] || !stepOb.y[yStepInd]) break;
                    var colBoundary = xArr.indexOf(+(ob.coords.x + stepOb.x[xStepInd]).toFixed(2)),
                        selectedResults = reconstructMatrixBySection(structure, currentRowInd, currentRowInd, currentColInd, colBoundary); // return only items from the row
                    selectedResults.splice(selectedResults.indexOf(ob), 1);
                }


                neighbor = chooseClosestNeighbor(selectedResults, ob)
                if (neighbor == undefined) { // this simply performs a check that will only pass if we are at the corner result that isn't supposed to have a neighbor up or right
                    break;
                }


                if (checkAgainstCriteria(srchCritOb, ob, neighbor, nodeInfo, structure)) {
                    closest.push(neighbor);

                }
                obCompTracker.push({
                    'focalPoint': ob, 'neighbors': closest.map(function (ele) {
                        return ele.coords.x + ' ' + ele.coords.y
                    })
                })
                break;
            case ('false,true'): //right col
                var rowBoundary = yArr.indexOf(+(ob.coords.y + stepOb.y[yStepInd]).toFixed(2)),
                    selectedResults = reconstructMatrixBySection(structure, currentRowInd, rowBoundary, currentColInd, currentColInd);
                selectedResults.splice(selectedResults.indexOf(ob), 1);

                while (selectedResults.length > 10 || selectedResults.length == 0 && counter < 30) { //inner while loop is to allow the step values arr to help adjust the amount of results we get back
                    ++counter;
                    if (rowBoundary == -1) ++yStepInd;
                    if (selectedResults.length > 10) {
                        ++yStepInd
                    }
                    if (!stepOb.x[xStepInd] || !stepOb.y[yStepInd]) break;
                    var rowBoundary = yArr.indexOf(+(ob.coords.y + stepOb.y[yStepInd]).toFixed(2)),
                        selectedResults = reconstructMatrixBySection(structure, currentRowInd, rowBoundary, currentColInd, currentColInd);
                    selectedResults.splice(selectedResults.indexOf(ob), 1);
                }

                neighbor = chooseClosestNeighbor(selectedResults, ob)
                if (neighbor == undefined) { // this simply performs a check that will only pass if we are at the corner result that isn't supposed to have a neighbor up or right
                    break;
                }


                if (checkAgainstCriteria(srchCritOb, ob, neighbor, nodeInfo, structure)) {
                    closest.push(neighbor);

                }

                obCompTracker.push({
                    'focalPoint': ob, 'neighbors': closest.map(function (ele) {
                        return ele.coords.x + ' ' + ele.coords.y
                    })
                })
                break;
            // don't worry about the top right dot, it has already recieved all the comparisons that it needs
        }
    }
    return obCompTracker
}









