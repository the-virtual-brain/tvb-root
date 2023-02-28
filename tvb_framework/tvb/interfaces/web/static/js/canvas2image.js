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

/* globals displayMessage */

var C2I_EXPORT_HEIGHT = 1080;
var canvasAndSvg=false;
/**
 * Function called on any visualizer, to export canvases into image/svg downloadable files..
 * @param kwargs an object with currently only one optional key:
 *   suggestedName: A name prefix for the figure
 */
function C2I_exportFigures(kwargs) {
    if ($("canvas, svg").filter(":visible").length === 0) {
        displayMessage("Invalid action. Please report to your TVB technical contact.", "errorMessage");
        return;
    }
    if ($("canvas").filter(":visible").length > 1 && $("svg").filter(":visible").length > 0) {
        canvasAndSvg = true;
    }
    if (canvasAndSvg) {
        addSnapshotCanvas();
        var main_canvas = document.getElementById("snapshotCanvas");
        main_canvas.style.visibility="visible";
        main_canvas.width = C2I_EXPORT_HEIGHT;
        main_canvas.height = C2I_EXPORT_HEIGHT;
    }
    $("canvas").filter(":visible").each(function () {
        if (canvasAndSvg) {
            __buildCanvas(this, main_canvas);
        }
        else {
            __storeCanvas(this, kwargs);
        }
    });

    let svgRef = $("svg").filter(":visible").filter(":not([display='none'])");
    if (svgRef.length > 1) {
        let filteredSVGs = [];
        let l = svgRef.length;
        for (let i = 0; i < l; i++) {
            let currentSVG = svgRef[i];
            let contained = false;
            for (let j = 0; j < l; j++) {
                let otherSVG = svgRef[j];
                if (i != j && $.contains(otherSVG, currentSVG)) {
                    contained = true;
                }
            }
            if (!contained) {
                filteredSVGs.push(currentSVG);
            }
        }
        svgRef = $(filteredSVGs);
    }

    svgRef.attr({ version: '1.1' , xmlns:"http://www.w3.org/2000/svg"});
    for(var i = 0; i < svgRef.length; i++) {
            if (i === svgRef.length - 1)
                __storeSVG(svgRef[i], kwargs, save = true);
            else {
                __storeSVG(svgRef[i], kwargs);
            }
    }
}

/**
 *This method save the svg html. Before this it also adds the required css styles.
 */
function __storeSVG(svgElement, kwargs, save) {
    // Wrap the svg element as to get the actual html and use that as the src for the image

    var wrap = document.createElement('div');
    wrap.appendChild(svgElement.cloneNode(true));
    var data = wrap.innerHTML;

    // get the styles for the svg

    $.get( deploy_context + "/static/style/subsection_svg.css", function (stylesheet) {
                                                                         // strip all
        var re = new RegExp("[\\s\\n^]*\\/\\*(.|[\\r\\n])*?\\*\\/" +     // block style comments
            "|([\\s\\n]*\\/\\/.*)" +                     // single line comments
            "|(^\\s*[\\r\\n])", "gm");                   // empty lines

        var svgStyle = "<defs><style type='text/css'><![CDATA["
            +  stylesheet.replace(re,"")
            + "]]></style></defs>";

        // embed the styles in svg
        var startingTag = data.substr(0, data.indexOf(">") + 1);
        var restOfSvg = data.substr(data.indexOf(">") + 1, data.length + 1);
        var styleAddedData = startingTag + svgStyle + restOfSvg;
        if(canvasAndSvg){
            var offsets = svgElement.getBoundingClientRect();
            var DOMURL = window.URL || window.webkitURL || window;
            var main_canvas = document.getElementById("snapshotCanvas");
            var ctx = main_canvas.getContext('2d');
            var img = new Image();
            var svg = new Blob([styleAddedData], {type: 'image/svg+xml'});
            var urlsvg = DOMURL.createObjectURL(svg);
            img.src = urlsvg;
            img.onload = function () {
                ctx.drawImage(img, offsets.left, offsets.top);
                DOMURL.revokeObjectURL(urlsvg);
                if(save){
                    __tryExport(main_canvas, kwargs, 25);
                }
            };
        }
        else {
            var url = '/project/figure/storeresultfigure/svg?';
            for (var k in kwargs) {
                url = url + k + '=' + kwargs[k];
            }

            // send it to server
            doAjaxCall({
                type: "POST", url: url,
                data: {"export_data": styleAddedData},
                success: function () {
                    displayMessage("Figure successfully saved!<br/> See Project section, Image archive sub-section.",
                        "infoMessage")
                },
                error: function () {
                    displayMessage("Could not store preview image, sorry!", "warningMessage")
                }
            });
        }
    } );
}

function C2IbuildUrlQueryString(baseUrl, kwargs){
    var ret = baseUrl;
    var i=0;
    for(var k in kwargs){
        if(kwargs.hasOwnProperty(k)){
            if(i == 0){
                ret += '?';
            }else{
                ret += '&';
            }
            ret +=  k + '=' + kwargs[k];
            i++;
        }
    }
    return ret;
}

/**
 * This function sends canvas' snapshot to server, after it has been prepared by <code>__storeCanvas()</code>
 *
 * NOTE: Use canvases <code>canvas.notReadyForExport</code> flag to indicate that their resize is not done
 * yet; if such flag exists, exporting continues only when it is set to <code>false</code> or after
 * <code>remainingTrials</code> trials
 *
 * @param canvas The **RESIZED** canvas whose snapshot is to be stored
 * @param kwargs To associate with current storage (e.g. suggestedName)
 * @param remainingTrials The number of times to poll for <code>canvas.notReadyForExport</code> flag
 * @private
 */
function __tryExport(canvas, kwargs, remainingTrials) {

    if (remainingTrials <= 0) {         // only try to export a limited number of times
        displayMessage("Could not export canvas data, sorry!", "warningMessage");
        return;
    }

    if (canvas.notReadyForExport) {
        // some canvases will set this flag to TRUE after they finish resizing, so they can be exported at Hi Res
        // undefined or FALSE means it CAN BE exported
        setTimeout(function () { __tryExport(canvas, kwargs, remainingTrials - 1); }, 300);
    } else {              // canvas is ready for export

        var data = canvas.toDataURL("image/png");

        if (data){       // don't store empty images
            var url = C2IbuildUrlQueryString('/project/figure/storeresultfigure/png', kwargs);

            doAjaxCall({
                type: "POST", url: url,
                data: {"export_data": data.replace('data:image/png;base64,', '')},
                success: function() {
                    displayMessage("Figure successfully saved!<br/> See Project section, " +
                                   "Image archive sub-section.", "infoMessage");
                } ,
                error: function() {
                    displayMessage("Could not store preview image, sorry!", "warningMessage");
                }
            });
        } else {            // there was no image data
            displayMessage("Canvas contains no image data. Try again or report to your TVB technical contact",
                           "warningMessage");
        }
        // restore original canvas size; non-webGL canvases (EEG, JIT) have custom resizing methods
        if (canvas.afterImageExport) {
            canvas.afterImageExport();
        }
    }
}

/**
 * This function deals with canvas storage. First it prepares it by calling its resize method
 * (<code>canvas.drawForImageExport</code>), then tries to save it
 * @param canvas The canvas whose image is to be stored
 * @param kwargs To associate with current storage (e.g. suggestedName)
 * @private
 */
function __storeCanvas(canvas, kwargs) {

    if (!canvas.drawForImageExport) {     // canvases which didn't set this method should not be saved
        return;
    }
    // If the canvas wishes to save more images it can define multipleImageExport
    // multipleImageExport receives a function that saves the current scene
    if (canvas.multipleImageExport){
        canvas.multipleImageExport(function(saveimgKwargs){
            $.extend(kwargs, saveimgKwargs);
            __tryExport(canvas, kwargs, 25);
        });
        return;
    }

    canvas.drawForImageExport();        // interface-like function that redraws the canvas at bigger dimension

    __tryExport(canvas, kwargs, 25);
}

function __buildCanvas(canvas, snapshotCanvas) {
    // if (!canvas.drawForImageExport) {     // canvases which didn't set this method should not be saved
    //     return;
    // }
    var ctx = snapshotCanvas.getContext('2d');
    offsets= canvas.getBoundingClientRect();
    // ctx.drawImage(canvas, canvas.offsetLeft, canvas.offsetTop, canvas.clientWidth, canvas.clientHeight);
    ctx.drawImage(canvas, offsets.left, offsets.top, canvas.clientWidth, canvas.clientHeight);
}

function addSnapshotCanvas() {
    if(document.getElementById('snapshotCanvas') === null) {
        var main_canvas = document.createElement('canvas');
        main_canvas.id = "snapshotCanvas";
        main_canvas.style.display = "none";
        var body = document.getElementsByTagName("body")[0];
        body.appendChild(main_canvas);
        main_canvas.drawForImageExport = function () {
            main_canvas.style.display = "block";
        };      // display
        main_canvas.afterImageExport = function () {
            main_canvas.style.visibility = "none";
        };     // hide
    }
}