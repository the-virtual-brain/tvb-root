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
 * This viewer was initiated as part of GSOC 2014 collaboration, by Robert Parcus
 */

/**
 * module volumeRPC responsible with fetching data from the server
 * It exposes a simple api via TSRPC_initNonStreaming, TSRPC_getViewAtTime
 * It also exposes a more complex api that prefetches and caches requests.
 * It is realized via TSRPC_initStreaming, TSRPC_startBuffering, TSRPC_stopBuffering, TSRPC_getViewAtTime
 * This last api requires a callback to the timeseriesVolume controller and it will start background processes.
 */
(function () {

    var tsRPC = {
        timeLength: 0,              // Number of timepoints in the Volume.

        bufferSize: 1,              // How many time points to get each time. It will be computed automatically, This is only the start value
        bufferL2Size: 1,            // How many sets of buffers can we keep at the same time in memory
        lookAhead: 10,             // How many sets of buffers should be loaded ahead of us each time?

        bufferL2: {},               // Contains all data from loaded views, limited by memory.
        urlVolumeData: "",          // Used to store the call for get_volume_view server function.

        urlBackgroundVolumeData: "",// The base url to the get_volume_view function of the background ts
        backgroundL2: {},           // Cache for the background. Only one entry Size == 1

        requestQueue: [],           // Used to avoid requesting a time point set while we are waiting for it.
        parserBlob: null,           // Used to store the JSON Parser Blob for web-workers.

        batchID: 0,                 // Used to ignore useless incoming ajax responses.
        streamToBufferID: null,     // ID from the buffering system's setInterval().
        getCurrentEntityAndTime: null, //  callback called by the streaming functions to get the selected entity and time point

        urlVoxelRegion: null        // url to fetch the name of the region that contains a voxel
    };

    function TSRPC_initNonStreaming(urlVolumeData, urlBackgroundVolumeData, urlVoxelRegion, entitySize) {
        tsRPC.urlVolumeData = urlVolumeData;
        tsRPC.urlBackgroundVolumeData = urlBackgroundVolumeData;
        tsRPC.urlVoxelRegion = urlVoxelRegion;
        tsRPC.timeLength = entitySize[3];
    }

    /**
     * @param urlVolumeData Url base for retrieving current slices data (for the left-side)
     * @param urlBackgroundVolumeData URL base for retrieving the T1 background image
     * @param entitySize The size of each plane
     * @param playbackRate
     * @param getCurrentEntityAndTime callback called by the streaming functions to get the selected entity and time point
     * The callback should return {currentTimePoint: , selectedEntity: }
     */
    function TSRPC_initStreaming(urlVolumeData, urlBackgroundVolumeData, entitySize, playbackRate, getCurrentEntityAndTime) {
        /**
         * This will be our JSON parser web-worker blob,
         * Using a webworker is a bit slower than parsing the jsons with
         * classical methods but it will prevent the main thread to be blocked
         * while waiting for the parsing, granting a smooth visualization.
         * We use this technique also to avoid writing a separate file
         * for each worker.
         */
        TSRPC_initNonStreaming(urlVolumeData, urlBackgroundVolumeData, null, entitySize);
        tsRPC.getCurrentEntityAndTime = getCurrentEntityAndTime;
        tsRPC.parserBlob = inlineWebWorkerWrapper(
            function () {
                self.addEventListener('message', function (e) {
                    // Parse JSON, send it to main thread, close the worker
                    self.postMessage(JSON.parse(e.data));
                    self.close();
                }, false);
            }
        );
        _setupBuffersSize(entitySize);
        // Fire the memory cleaning procedure
        window.setInterval(freeBuffer, playbackRate * 20);
    }

    /**
     * Automatically determine optimal bufferSizer, depending on data dimensions.
     */
    function _setupBuffersSize(entitySize) {
        let tpSize = Math.max(entitySize[0], entitySize[1], entitySize[2]);
        tpSize = tpSize * tpSize;
        //enough to avoid waisting bandwidth and to parse the json smoothly
        while (tsRPC.bufferSize * tpSize <= 50000) {
            tsRPC.bufferSize++;
        }
        //Very safe measure to avoid crashes. Tested on Chrome.
        while (tsRPC.bufferSize * tpSize * tsRPC.bufferL2Size <= 157286400) {
            tsRPC.bufferL2Size *= 2;
        }
        tsRPC.bufferL2Size /= 2;
    }

    /**
     * Requests file data without blocking the main thread if possible.
     * @param fileName The target URL or our request
     * @param sect The section index of the wanted data in our buffering system.
     */
    function asyncRequest(fileName, sect) {
        const index = tsRPC.requestQueue.indexOf(sect);
        const privateID = tsRPC.batchID;

        if (index < 0) {
            tsRPC.requestQueue.push(sect);
            doAjaxCall({
                async: true,
                url: fileName,
                method: "POST",
                mimeType: "text/plain",
                success: function (response) {
                    if (privateID === tsRPC.batchID) {
                        parseAsync(response, function (json) {
                            // save the parsed JSON
                            tsRPC.bufferL2[sect] = json;
                            const idx = tsRPC.requestQueue.indexOf(sect);
                            if (idx > -1) {
                                tsRPC.requestQueue.splice(idx, 1);
                            }
                        });
                    }
                },
                error: function () {
                    displayMessage("Could not retrieve data from the server!", "warningMessage");
                }
            });
        }
    }

    /**
     * Build a worker from an anonymous function body. Returns and URL Blob
     * @param workerBody The anonymous function to convert into URL BLOB
     * @returns URL Blob that can be used to invoke a web worker
     */
    function inlineWebWorkerWrapper(workerBody) {
        return URL.createObjectURL(
            new Blob(['(', workerBody.toString(), ')()'], {type: 'application/javascript'})
        );
    }

    /**
     * Parses JSON data in a web-worker. Has a fall back to traditional parsing.
     * @param data The json data to be parsed
     * @param callback Function to be called after the parsing
     */
    function parseAsync(data, callback) {
        let worker;
        let json;
        if (window.Worker) {
            worker = new Worker(tsRPC.parserBlob);
            worker.addEventListener('message', function (e) {
                json = e.data;
                callback(json);
            }, false);
            worker.postMessage(data);
        } else {
            json = JSON.parse(data);
            callback(json);
        }
    }

    function voxelToUrlFragment(selectedEntity) {
        const xPlane = ";x_plane=" + selectedEntity[0];
        const yPlane = ";y_plane=" + selectedEntity[1];
        const zPlane = ";z_plane=" + selectedEntity[2];
        return xPlane + yPlane + zPlane;
    }

    function buildRequestUrl(urlVolumeData, from, to, selectedEntity) {
        const spacialFragment = voxelToUrlFragment(selectedEntity);
        const modeSv = ";mode=" + selectedEntity[3] + ";var=" + selectedEntity[4];
        return urlVolumeData + "from_idx=" + from + ";to_idx=" + to + spacialFragment + modeSv;
    }

    /**
     *  This function is called whenever we can, to load some data ahead of
     *  were we're looking.
     */
    function streamToBuffer() {
        // we avoid having too many requests at the same time
        if (tsRPC.requestQueue.length < 2) {
            const point = tsRPC.getCurrentEntityAndTime();
            const currentSection = Math.ceil(point.currentTimePoint / tsRPC.bufferSize);
            const maxSections = Math.floor(tsRPC.timeLength / tsRPC.bufferSize);

            for (let i = 0; i <= tsRPC.lookAhead && i < maxSections; i++) {
                const toBufferSection = Math.min(currentSection + i, maxSections);
                // If not already requested:
                if (!tsRPC.bufferL2[toBufferSection] && tsRPC.requestQueue.indexOf(toBufferSection) < 0) {
                    const from = toBufferSection * tsRPC.bufferSize;
                    const to = Math.min(from + tsRPC.bufferSize, tsRPC.timeLength);
                    const query = buildRequestUrl(tsRPC.urlVolumeData, from, to, point.selectedEntity);
                    asyncRequest(query, toBufferSection);
                    return; // break out of the loop
                }
            }
        }
    }

    /**
     *  This function is called to erase some elements from bufferL2 array and avoid
     *  consuming too much memory.
     */
    function freeBuffer() {
        const point = tsRPC.getCurrentEntityAndTime();
        const section = Math.floor(point.currentTimePoint / tsRPC.bufferSize);
        const bufferedElements = Object.keys(tsRPC.bufferL2).length;

        if (bufferedElements > tsRPC.bufferL2Size) {
            for (let idx in tsRPC.bufferL2) {
                if (idx < (section - tsRPC.bufferL2Size / 2) % tsRPC.timeLength || idx > (section + tsRPC.bufferL2Size / 2) % tsRPC.timeLength) {
                    delete tsRPC.bufferL2[idx];
                }
            }
        }
    }

    function _getViewAtTimeNoCache(t, selectedEntity, urlVolumeData) {
        const from = t;
        const to = Math.min(1 + t, tsRPC.timeLength);
        const query = buildRequestUrl(urlVolumeData, from, to, selectedEntity);

        const buffer = HLPR_readJSONfromFile(query);
        return [buffer[0][0], buffer[1][0], buffer[2][0]];
    }

    /**
     * Gets volumetric slices similar to TSRPC_getViewAtTime.
     * But this function has no temporal buffering support and it uses a different caching strategy.
     * It caches just the last request.
     * This behaviour is appropriate for the background.
     * It also uses a different url to fetch data.
     */
    function TSRPC_getBackgroundView(selectedEntity) {
        const key = selectedEntity + "";
        if (tsRPC.backgroundL2[key]) {
            return tsRPC.backgroundL2[key];
        } else {
            const ret = _getViewAtTimeNoCache(0, selectedEntity, tsRPC.urlBackgroundVolumeData);
            tsRPC.backgroundL2 = {};
            tsRPC.backgroundL2[key] = ret;
            return ret;
        }
    }

    /**
     *  This functions returns the X,Y,Z data from time-point t.
     * @param t The time point we want to get
     * @param selectedEntity The selected voxel
     * @returns Array with only the data from the x,y,z plane at time-point t.
     */
    function TSRPC_getViewAtTime(t, selectedEntity) {
        const section = Math.floor(t / tsRPC.bufferSize);
        // Note that the cache key is time. The voxel is not part of the key.
        // This cache will be invalidated if another voxel is selected by calling TSRPC_startBuffering.
        if (tsRPC.bufferL2[section]) { // We have that slice in memory
            const buffer = tsRPC.bufferL2[section];
            t = t % tsRPC.bufferSize;
            return [buffer[0][t], buffer[1][t], buffer[2][t]];
        } else { // We need to load that slice from the server
            return _getViewAtTimeNoCache(t, selectedEntity, tsRPC.urlVolumeData);
        }
    }

    function TSRPC_startBuffering() {
        // Only start buffering id the computed buffer length > 1. Whe only 1 step can be retrieved it is not worthy,
        // and we will have duplicate retrievals generated
        if (!tsRPC.streamToBufferID && tsRPC.bufferSize > 1) {
            tsRPC.batchID++;
            tsRPC.requestQueue = [];
            tsRPC.bufferL2 = {};
            tsRPC.streamToBufferID = window.setInterval(streamToBuffer, 0);
        }
    }

    function TSRPC_stopBuffering() {
        window.clearInterval(tsRPC.streamToBufferID);
        tsRPC.streamToBufferID = null;
    }

    function TSRPC_getVoxelRegion(selectedEntity, onSuccess) {
        if (!tsRPC.urlVoxelRegion) {
            onSuccess('unknown');
            return;
        }
        doAjaxCall({
            url: tsRPC.urlVoxelRegion + voxelToUrlFragment(selectedEntity),
            success: onSuccess
        });
    }


    window.TSRPC_initNonStreaming = TSRPC_initNonStreaming;
    window.TSRPC_initStreaming = TSRPC_initStreaming;
    window.TSRPC_getViewAtTime = TSRPC_getViewAtTime;
    window.TSRPC_startBuffering = TSRPC_startBuffering;
    window.TSRPC_stopBuffering = TSRPC_stopBuffering;
    window.TSRPC_getVoxelRegion = TSRPC_getVoxelRegion;
    window.TSRPC_getBackgroundView = TSRPC_getBackgroundView;

    window._debug_tsRPC = tsRPC;

})();
