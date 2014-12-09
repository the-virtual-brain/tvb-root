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

/* globals displayMessage */

var C2I_EXPORT_HEIGHT = 1080;

/**
 * Function called on any visualizer, to export canvases into image/svg downloadable files..
 * @param kwargs an object with 2 optional keys
 *   operationId  : The operation which created the figure
 *   suggestedName: A name for the figure
 */
function C2I_exportFigures(kwargs) {
    if ($("canvas, svg").filter(":visible").length === 0) {
        displayMessage("Invalid action. Please report to your TVB technical contact.", "errorMessage");
        return;
    }
    $("canvas").filter(":visible").each(function () {
        __storeCanvas(this, kwargs);
    });

    var svgRef = $("svg").filter(":visible");
    svgRef.attr({ version: '1.1' , xmlns:"http://www.w3.org/2000/svg"});
    svgRef.each(function () {
        __storeSVG(this, kwargs);
    });
}

/**
 *This method save the svg html. Before this it also adds the required css styles.
 */
function __storeSVG(svgElement, kwargs) {
	// Wrap the svg element as to get the actual html and use that as the src for the image

	var wrap = document.createElement('div');
	wrap.appendChild(svgElement.cloneNode(true));
	var data = wrap.innerHTML;

    // get the styles for the svg
    $.get( "/static/style/sections_svg.css", function (stylesheet) {
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

        var url = '/project/figure/storeresultfigure/svg?';
        for(var k in kwargs){
            url = url + k + '=' + kwargs[k];
        }

        // send it to server
        doAjaxCall({  type: "POST", url: url,
            data: {"export_data": styleAddedData},
            success: function() {
                displayMessage("Figure successfully saved!<br/> See Project section, Image archive sub-section.",
                               "infoMessage")
            } ,
            error: function() {
                displayMessage("Could not store preview image, sorry!", "warningMessage")
            }
        });

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
 * NOTE: Some canvases (e.g. MPLH5) set <code>canvas.notReadyForExport</code> flag to indicate that their resize is not done
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
        // the mplh5 canvases will set this flag to TRUE after they finish resizing, so they can be exported at Hi Res
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
        // restore original canvas size; non-webGL canvases (EEG, mplh5, JIT) have custom resizing methods
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