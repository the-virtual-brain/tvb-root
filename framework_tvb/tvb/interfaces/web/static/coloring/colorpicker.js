var TVBUI = TVBUI || {};

/**
 * @module simple color picker
 */
(function() {
    "use strict";
    var TILE_SIZE = 28;

    function setBackground(el, col){
        el.css('background-color', "rgb(" + col[0] + "," + col[1] + "," + col[2] + ")");
    }

    /**
     * Creates a tiled color picker using the given set of colors
     * This component assumes a dom structure, it only creates the color tiles.
     * It also assumes that menu showing/hiding is handled externally by genericTVB.js:setupMenuEvents
     * @param selector Selector to the <nav> element.
     * @param colors An array of colors represented as 3 element arrays of ints 0..255
     * @constructor
     */
    function ColorTilePicker(selector, colors){
        var self = this;
        this.colors = colors;
        this.color = colors[0];

        var dom = $(selector);
        var ul = dom.find('ul');
        this.previewBox = dom.find('.colorTilePickerSelected');

        setBackground(this.previewBox, this.color);
        this._sizeContainer(dom.find('.dropdown-pane'));
        ul.click(function(ev){ self._onClick(ev.target);});
        this._addTiles(ul);
    }

    ColorTilePicker.prototype._onClick = function(li){
        var idx = $(li).index();
        this.color = this.colors[idx];
        setBackground(this.previewBox, this.color);
    };

    ColorTilePicker.prototype._sizeContainer = function(container){
        var cols = Math.ceil(Math.sqrt(this.colors.length));
        var rows = Math.ceil(this.colors.length / cols);
        container.width(TILE_SIZE * cols).height(TILE_SIZE * rows);
    };

    ColorTilePicker.prototype._addTiles = function(ul){
        for(var i = 0; i < this.colors.length; i++){
            var li = $('<li>').width(TILE_SIZE).height(TILE_SIZE);
            setBackground(li, this.colors[i]);
            li.appendTo(ul);
        }
    };

    TVBUI.ColorTilePicker = ColorTilePicker;
})();