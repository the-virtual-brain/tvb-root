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

/* globals displayMessage, ColSch_initColorSchemeGUI, ColSch_getAbsoluteGradientColorString,
            doAjaxCall, d3, HLPR_readJSONfromFile,
            TSV_initVolumeView, TSV_drawVolumeScene, TSV_hitTest, TSV_getQuadrant,
            TSF_drawGraphs, TSF_updateTSFragment, TSF_initVisualizer, TSF_updateTimeGauge,
            TSRPC_initNonStreaming, TSRPC_initStreaming, TSRPC_getViewAtTime,
            TSRPC_startBuffering, TSRPC_stopBuffering
             */
(function () { // module timeseriesVolume controller
// ==================================== INITIALIZATION CODE START ===========================================
    var tsVol = {
        selectedEntity: [0, 0, 0, 0, 0],  // The selected voxel; [i, j, k, mode, sv].
        entitySize: [0, 0, 0],      // The size of each plane
        selectedQuad: 0,            // The quadrant selected by the user every time
        timeLength: 0,              // Number of timepoints in the Volume.
        currentTimePoint: 0,
        playbackRate: 66,           // This is a not acurate lower limit for playback speed.
        playerIntervalID: null,     // ID from the player's setInterval().
        data: {},                   // The actual data to be drawn to canvas.
        dataSize: "",               // Used first to contain the file ID and then it's dimension.
        slidersClicked: false,      // Used to handle the status of the sliders.
        samplePeriod: 0,            // Meta data. The sampling period of the time series
        samplePeriodUnit: "",       // Meta data. The time unit of the sample period
        backgroundColorScale: null, // Color scale for the anatomical background
        haveBackground: false
    };

    var SLIDERS = ["X", "Y", "Z"];
    var SLIDERIDS = ["sliderForAxisX", "sliderForAxisY", "sliderForAxisZ"];

    /** Initializes all state related to the volumetric part */
    function init_VolumeController(volumeShape) {
        tsVol.dataSize = $.parseJSON(volumeShape);
        // set the center entity as the selected one
        tsVol.selectedEntity[0] = Math.floor(tsVol.dataSize[1] / 2);
        tsVol.selectedEntity[1] = Math.floor(tsVol.dataSize[2] / 2);
        tsVol.selectedEntity[2] = Math.floor(tsVol.dataSize[3] / 2);

        // get entities number of voxels
        tsVol.entitySize[0] = tsVol.dataSize[1];
        tsVol.entitySize[1] = tsVol.dataSize[2];
        tsVol.entitySize[2] = tsVol.dataSize[3];
        // get entities number of time points
        tsVol.entitySize[3] = tsVol.dataSize[0];           //Number of time points;
        tsVol.timeLength = tsVol.dataSize[0];
    }

    /**
     * Initializes selection of mode and state-variable
     */
    function _initModeAndStateVariable() {
        const modeSelector = TVBUI.modeAndStateSelector("#channelSelector", 0);
        modeSelector.modeChanged(function (id, val) {
            tsVol.selectedEntity[3] = parseInt(val);
            drawSceneFunctional(tsVol.currentTimePoint);
        });
        modeSelector.stateVariableChanged(function (id, val) {
            tsVol.selectedEntity[4] = parseInt(val);
            drawSceneFunctional(tsVol.currentTimePoint);
        });
    }

    function TSV_startVolumeStaticVisualizer(urlVolumeData, urlVoxelRegion, minValue, maxValue,
                                             volumeShape, volOrigin, sizeOfVoxel,
                                             urlBackgroundVolumeData, minBackgroundValue, maxBackgroundValue) {
        init_VolumeController(volumeShape);
        TSV_initVolumeView(tsVol.dataSize, minValue, maxValue, $.parseJSON(sizeOfVoxel), $.parseJSON(volOrigin)[0]);

        tsVol.selectedQuad = TSV_getQuadrant(0);

        TSRPC_initNonStreaming(urlVolumeData, urlBackgroundVolumeData, urlVoxelRegion, tsVol.entitySize);

        ColSch_initColorSchemeGUI(minValue, maxValue, function () {
            drawVolumeScene(tsVol.currentTimePoint);
        });

        tsVol.backgroundColorScale = new ColorScale(minBackgroundValue, maxBackgroundValue, 'Grays');
        tsVol.haveBackground = urlBackgroundVolumeData !== '';
        ColSch.colorScale = new AlphaClampColorScale(minValue, maxValue, ColSch.colorScale._colorSchemeName, 255, 0.0, 1.0);

        $("#canvasVolumes").mousedown(onVolumeMouseDown).mouseup(updateSelectedVoxelInfo);

        startPositionSliders({
            change: _coreMoveSliderAxis,
            slide: function (event, ui) {
                _coreMoveSliderAxis(event, ui);
                drawVolumeScene(tsVol.currentTimePoint);
            }
        });
        drawVolumeScene(tsVol.currentTimePoint);
        updateSelectedVoxelInfo();
    }

    /**
     * Make all the necessary initialisations and draws the default view, with the center voxel selected
     * @param urlVolumeData          Url base for retrieving current slices data (for the left-side)
     * @param urlTimeSeriesData      URL base for retrieving TS (right side)
     * @param minValue          The minimum value for all the slices
     * @param maxValue          The maximum value for all the slices
     * @param samplePeriod      Number representing how frequent the signal was being measured
     * @param samplePeriodUnit  Unit-Measure for the sampling period (e.g. ms, s)
     * @param volumeShape       Json with the shape of the full TS
     * @param volOrigin         The origin of the rendering; irrelevant in 2D, for now
     * @param sizeOfVoxel       How the voxel is sized on each axis; [xScale, yScale, zScale]
     * @param urlBackgroundVolumeData T1 background
     * @param minBackgroundValue Min gray
     * @param maxBackgroundValue Max gray
     */
    function TSV_startVolumeTimeSeriesVisualizer(urlVolumeData, urlTimeSeriesData, minValue, maxValue,
                                                 samplePeriod, samplePeriodUnit, volumeShape, volOrigin, sizeOfVoxel,
                                                 urlBackgroundVolumeData, minBackgroundValue, maxBackgroundValue) {

        init_VolumeController(volumeShape);
        TSV_initVolumeView(tsVol.dataSize, minValue, maxValue, $.parseJSON(sizeOfVoxel), $.parseJSON(volOrigin)[0]);

        tsVol.selectedQuad = TSV_getQuadrant(0);

        tsVol.samplePeriod = samplePeriod;
        tsVol.samplePeriodUnit = samplePeriodUnit;

        TSRPC_initStreaming(urlVolumeData, urlBackgroundVolumeData, tsVol.entitySize, tsVol.playbackRate, function () {
            return {currentTimePoint: tsVol.currentTimePoint, selectedEntity: tsVol.selectedEntity};
        });

        ColSch_initColorSchemeGUI(minValue, maxValue, function () {
            drawSceneFunctional(tsVol.currentTimePoint);
        });

        tsVol.backgroundColorScale = new ColorScale(minBackgroundValue, maxBackgroundValue, 'Grays');
        tsVol.haveBackground = urlBackgroundVolumeData !== '';
        ColSch.colorScale = new AlphaClampColorScale(minValue, maxValue, ColSch.colorScale._colorSchemeName, 255, 0.0, 1.0);

        // Update the data shared with the SVG Time Series Fragment
        TSF_updateTSFragment(tsVol.selectedEntity, tsVol.currentTimePoint);

        // Start the SVG Time Series Fragment and draw it.
        TSF_initVisualizer(urlTimeSeriesData, tsVol.timeLength, tsVol.samplePeriod,
            tsVol.samplePeriodUnit, minValue, maxValue);
        TSF_drawGraphs();

        $("#canvasVolumes").mousedown(customMouseDown).mouseup(customMouseUp);
        initTimeControls();
        startPositionSliders({change: slideMoved, slide: slideMove});
        startMovieSlider();

        drawSceneFunctional(tsVol.currentTimePoint);
        _initModeAndStateVariable();
    }

// ==================================== INITIALIZATION CODE END =============================================

    /**
     * Draws the current view depending on the selected entity
     * @param tIndex The time point we want to draw
     */
    function drawSceneFunctional(tIndex) {
        // if we pass no tIndex the function will play
        // from the tsVol.currentTimePoint incrementing it by 1 or going back to 0.
        if (tIndex === undefined) {
            tIndex = tsVol.currentTimePoint;
            tsVol.currentTimePoint++;
            tsVol.currentTimePoint = tsVol.currentTimePoint % tsVol.timeLength;
        }
        TSF_updateTSFragment(tsVol.selectedEntity, tsVol.currentTimePoint);
        drawVolumeScene(tIndex);
        updateMoviePlayerSlider();
    }

    function drawVolumeScene(tIndex) {
        // An array containing the view for each plane.
        let sliceArray = TSRPC_getViewAtTime(tIndex, tsVol.selectedEntity);
        let layers = [{sliceArray: sliceArray, colorScale: ColSch.colorScale}];

        if (tsVol.haveBackground) {
            let backgroundSliceArray = TSRPC_getBackgroundView(tsVol.selectedEntity);
            layers.splice(0, 0, {sliceArray: backgroundSliceArray, colorScale: tsVol.backgroundColorScale});
        }

        TSV_drawVolumeScene(layers, tsVol.selectedEntity);
    }

// ==================================== PICKING RELATED CODE START ==========================================

    /** Updates the selected quad and entity. and the sliders */
    function onVolumeMouseDown(e) {
        const hit = TSV_hitTest(e);
        if (!hit) {
            return;
        }
        tsVol.selectedQuad = hit.selectedQuad;
        tsVol.selectedEntity[tsVol.selectedQuad.axes.x] = hit.selectedEntityOnX;
        tsVol.selectedEntity[tsVol.selectedQuad.axes.y] = hit.selectedEntityOnY;
        updateSliders();
        drawVolumeScene(tsVol.currentTimePoint);
    }

    function customMouseDown(e) {
        e.preventDefault();
        this.mouseDown = true;            // `this` is the canvas

        // Implements picking and redraws the scene. Updates sliders too.
        if (tsVol.playerIntervalID) {
            stopPlayback();
            tsVol.resumePlayer = true;
        }
        onVolumeMouseDown(e);
        TSF_updateTSFragment(tsVol.selectedEntity, tsVol.currentTimePoint);
        updateMoviePlayerSlider();
    }

    function customMouseUp(e) {
        e.preventDefault();
        this.mouseDown = false;

        if (tsVol.resumePlayer) {
            window.setTimeout(playBack, tsVol.playbackRate * 2);
            tsVol.resumePlayer = false;
        }
        if (tsVol.selectedQuad.index === 3) {
            TSF_drawGraphs();
        }
    }

// ==================================== PICKING RELATED CODE  END  ==========================================

// ==================================== UI RELATED CODE START ===============================================

    function initTimeControls() {
        $('#btnSeekFirst').click(seekFirst);
        $('#btnPlayPreviousTimePoint').click(playPreviousTimePoint);
        $('#btnPlay').click(togglePlayback);
        $('#btnPlayNextTimePoint').click(playNextTimePoint);
        $('#btnSeekEnd').click(seekEnd);
    }

    /**
     * Code for the navigation slider. Creates the x,y,z sliders and adds labels
     */
    function startPositionSliders(options) {
        for (let i = 0; i < 3; i++) {
            const value = tsVol.selectedEntity[i];
            const opts = {
                value: value,
                min: 0,
                max: tsVol.entitySize[i] - 1, // yeah.. if we start from zero we need to subtract 1
                animate: true,
                orientation: "horizontal",
                change: options.change,// call this function *after* the slide is moved OR the value changes
                slide: options.slide //  we use this to keep it smooth.
            };
            $("#sliderForAxis" + SLIDERS[i]).slider(opts);
            $("#labelCurrentValueAxis" + SLIDERS[i]).empty().text("[" + value + "]");
            $("#labelMaxValueAxis" + SLIDERS[i]).empty().text(opts.max);
        }
    }

    /**
     * Code for "movie player" slider. Creates the slider and adds labels
     */
    function startMovieSlider() {
        let value = 0;
        const opts = {
            value: value,
            min: 0,
            max: tsVol.timeLength - 1,
            animate: true,
            orientation: "horizontal",
            range: "min",
            stop: moviePlayerMoveEnd,
            slide: moviePlayerMove
        };

        $("#movieSlider").slider(opts);

        const actualTime = value * tsVol.samplePeriod;
        const totalTime = (tsVol.timeLength - 1) * tsVol.samplePeriod;
        $("#labelCurrentTimeStep").empty().text("[" + actualTime.toFixed(2) + "]");
        $("#labelMaxTimeStep").empty().text(totalTime.toFixed(2) + " (" + tsVol.samplePeriodUnit + ")");
    }

// ==================================== CALLBACK FUNCTIONS START ===============================================

    function playBack() {
        if (!tsVol.playerIntervalID) {
            tsVol.playerIntervalID = window.setInterval(drawSceneFunctional, tsVol.playbackRate);
        }
        $("#btnPlay").attr("class", "action action-pause");
        TSRPC_startBuffering();
    }

    function stopPlayback() {
        window.clearInterval(tsVol.playerIntervalID);
        tsVol.playerIntervalID = null;
        $("#btnPlay").attr("class", "action action-run");
        TSRPC_stopBuffering();
    }

    function togglePlayback() {
        if (!tsVol.playerIntervalID) {
            playBack();
        } else {
            stopPlayback();
        }
    }

    function playNextTimePoint() {
        tsVol.currentTimePoint++;
        tsVol.currentTimePoint = tsVol.currentTimePoint % (tsVol.timeLength);
        drawSceneFunctional(tsVol.currentTimePoint);
    }

    function playPreviousTimePoint() {
        if (tsVol.currentTimePoint === 0) {
            tsVol.currentTimePoint = tsVol.timeLength;
        }
        drawSceneFunctional(--tsVol.currentTimePoint);
    }

    function seekFirst() {
        tsVol.currentTimePoint = 0;
        drawSceneFunctional(tsVol.currentTimePoint);
    }

    function seekEnd() {
        tsVol.currentTimePoint = tsVol.timeLength - 1;
        drawSceneFunctional(tsVol.currentTimePoint - 1);
    }

    /**
     * Updates the position and values of the x,y,z navigation sliders when we click the canvas.
     */
    function updateSliders() {
        for (let i = 0; i < 3; i++) {
            $("#sliderForAxis" + SLIDERS[i]).slider("option", "value", tsVol.selectedEntity[i]); //Update the slider value
            $('#labelCurrentValueAxis' + SLIDERS[i]).empty().text('[' + tsVol.selectedEntity[i] + ']'); //update label
        }
    }

    /**
     * While the navigation sliders are moved, this redraws the scene accordingly.
     */
    function slideMove(event, ui) {
        if (tsVol.playerIntervalID) {
            stopPlayback();
            tsVol.resumePlayer = true;
        }
        tsVol.slidersClicked = true;
        _coreMoveSliderAxis(event, ui);
        drawSceneFunctional(tsVol.currentTimePoint);
    }

    /**
     * After the navigation sliders are changed, this redraws the scene accordingly.
     */
    function slideMoved(event, ui) {
        if (tsVol.slidersClicked) {
            tsVol.slidersClicked = false;

            if (tsVol.resumePlayer) {
                tsVol.resumePlayer = false;
                window.setTimeout(playBack, tsVol.playbackRate * 2);
            }
        }
        _coreMoveSliderAxis(event, ui);
        TSF_updateTSFragment(tsVol.selectedEntity, tsVol.currentTimePoint);
    }

    function _coreMoveSliderAxis(event, ui) {
        const quadID = SLIDERIDS.indexOf(event.target.id);
        const selectedQuad = TSV_getQuadrant([quadID]);

        //  Updates the label value on the slider.
        $("#labelCurrentValueAxis" + SLIDERS[quadID]).empty().text('[' + ui.value + ']');
        //  Setup things to draw the scene pointing to the right voxel and redraw it.
        if (quadID === 1) {
            tsVol.selectedEntity[selectedQuad.axes.x] = ui.value;
        } else {
            tsVol.selectedEntity[selectedQuad.axes.y] = ui.value;
        }
    }

    /**
     * Updated the player slider bar while playback is on.
     */
    function updateMoviePlayerSlider() {
        _coreUpdateMovieSlider(tsVol.currentTimePoint, true);
    }

    /**
     * Updates the value at the end of the player bar when we move the handle.
     */
    function moviePlayerMove(event, ui) {
        _coreUpdateMovieSlider(ui.value, false);
    }

    function _coreUpdateMovieSlider(timePoint, updateSlider) {
        if (updateSlider) {
            $("#movieSlider").slider("option", "value", tsVol.currentTimePoint);
        }
        const actualTime = timePoint * tsVol.samplePeriod;
        $('#labelCurrentTimeStep').empty().text("[" + actualTime.toFixed(2) + "]");
        TSF_updateTimeGauge(timePoint);
    }

    /*
    * Redraws the scene at the selected time-point at the end of a slide action.
    * Calling this during the whole slide showed to be too expensive, so the
    * new time-point is drawn only when the user releases the click from the handler
    */
    function moviePlayerMoveEnd(event, ui) {
        tsVol.currentTimePoint = ui.value;
        drawSceneFunctional(tsVol.currentTimePoint);
    }

    function updateSelectedVoxelInfo() {
        TSRPC_getVoxelRegion(tsVol.selectedEntity, function (response) {
            $('#voxelRegionLabel').text(response);
        });
    }

// ==================================== CALLBACK FUNCTIONS END ===============================================
// ==================================== UI RELATED CODE END ==================================================

// module exports
    window.TSV_startVolumeTimeSeriesVisualizer = TSV_startVolumeTimeSeriesVisualizer;
    window.TSV_startVolumeStaticVisualizer = TSV_startVolumeStaticVisualizer;
    window._debug_tsVol = tsVol;
})();