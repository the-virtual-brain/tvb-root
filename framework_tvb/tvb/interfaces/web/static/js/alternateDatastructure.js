/**
 * this file is mean't to house the API skeleton for the new PSE datastructure. Ideally there needs to be several central features however they end up being coded.
 * --support for the addition of new results into the datastructure, and eventually onto the canvas
 * --support for iterative searching over each result and comparing its metrics to some user filter criteria
 * --support for iterative comparison of results and their neighbors within a certain distance, and determining whether the difference between the metrics of the compared results is above or below a user input criteria
 * ----this may include a method of grabbing all of the nearby results within some geometric area (circle or square) based on the step values that have been used to generate the different sets of results in the datagroup.
 *
 * Created by dev on 7/14/16.
 */

//this function will take in the original 1 dimensional array (seriesArray likely) and generate a matrix (sparse or condensed) that will be accessed upon plotting, filtering, and generating overlays.
// **I might be leaning towards using a CSR (compressed sparse Row) structure here to actually accomplish the reasonable sized storing**
function createStructure(data, xlabels, ylabels) {

}

//very important to be able to add new results into the current structure whatever winds up being selected
//this could be a good place to update a list that is storing the step values that have been used so far. Probably the call to update the coordinate arrays
function mergeResults(newData, data) {

    //this function will be necessary because there will need to be a single array comprised of the parameter values used to generate the results.
    // Think of arrays of the xparameter and yparameter values used as coordinates to position results in the canvas; these arrays need updating when more results are added.
    function updateCoordinateArrays(oldArr, NewArr) {

    }

}



//this function will gather information about the specifications that the user has selected above, and the logical relation between them (AND,OR) and will iteratively progress through the results one by one adding them to a return if they fit the search criteria
function filterResults(data) {


}

//this function will calculate the difference between each result and its neighbors upon a selected metric. This difference will be compared to a user input value, and will dictate whether there are colored contour lines added inbetween results to help visualize the change in results based on parameter change.
//don't forget that this will need to have some sort of threshold for the number of results that are coming back from the geometric selection before proceeding
//      if there happens to be too many decrease the distance value to the next smallest step value (x&y step arrays) that exists from the calculations.
function compareToNeighbors(data, xSteps, ySteps) {

    //this function is a helper for compareToNeighbors which will generate an object composed of other results within a specific distance from the result of interest (one currently being investigated out of all).
    //the current idea here is to go with a circular geometric area from which we will select all the results within the radius distance from the central result.
    //the other option which might eventually be easier is to try to use a square shape in the parameter space.***I think this is what I will lean towards***
    function geometricSelection(data, centralResult, distance) {

    }

    //this function will take whatever number of results have been selected within the distance from the focusObj (result under investigation), and will now attach distance values to each object in the geometric selection specific to the x or y parameter
    function assignDistance(focusObj, geoSelected) {

    }

}









