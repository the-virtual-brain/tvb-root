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

/**
 * This method allows you to make zoom on a canvas. The zoomed img will be removed on double click.
 *
 * @param sourceCanvasId the id of the canvas on which you want to make zoom
 * @param destinationDivId the id of the div in which you want to be displayed the zoomed img
 * @param zoomFactor the size of the zoom
 */
function zoom(sourceCanvasId, destinationDivId, zoomFactor) {
    var sourceCanvas = document.getElementById(sourceCanvasId);
    var dataUrl = sourceCanvas.toDataURL();
    var imgId = 'zoomedImg';

    var oldImg = $('#' + imgId);
    if (oldImg != null || oldImg != undefined) {
        oldImg.remove();
    }

    var img = document.createElement('img');
    img.id = imgId;
    img.style.border = '3px solid gray';
    img.style.width = sourceCanvas.width + zoomFactor + 'px';
    img.style.height = sourceCanvas.height + zoomFactor + 'px';
    img.style.position = 'absolute';
    img.style.left = 0 + 'px';
    img.style.top = 0 + 'px';
    img.style.backgroundColor = 'white';
    img.style.cursor = 'move';
    //img.style.zIndex = 99;
    img.src = dataUrl;

    document.getElementById(destinationDivId).appendChild(img);
    
    $("#" + imgId).draggable();
    removeElementOnDbClick(img.id);
}

function removeElementOnDbClick(elementId) {
    $('#' + elementId).dblclick(function() {
        $('#' + elementId).remove();
    });
}

function displaySection(sourceCanvasId, destinationDivId, axis, firstTime) {
    var sourceCanvas = document.getElementById(sourceCanvasId);
    var dataUrl = sourceCanvas.toDataURL();
    var imgId = 'zoomedImg' + '_' + axis;
    
    var oldImg = $('#' + imgId);
    if (oldImg != null || oldImg != undefined) {
        oldImg.remove();
    }

    var img = document.createElement('img');
    img.id = imgId;
    img.style.width = 250 + 'px';
    img.style.height = 172 + 'px';
    img.style.right = 0 + 'px';
    img.style.top = 0 + 'px';
    img.style.float = "right";
    img.style.position = "absolute";
    if (firstTime) {
        img.style.backgroundColor = 'white';
    } else {
        img.style.backgroundColor = 'black';
    }
    img.src = dataUrl;
    if (document.getElementById(destinationDivId)) {
    	document.getElementById(destinationDivId).appendChild(img);
    }
}
