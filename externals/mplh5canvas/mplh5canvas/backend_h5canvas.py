"""An HTML5 Canvas backend for matplotlib.

Simon Ratcliffe (sratcliffe@ska.ac.za)
Ludwig Schwardt (ludwig@ska.ac.za)

Copyright (c) 2010, SKA South Africa
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.
Neither the name of SKA South Africa nor the names of its contributors may be used to endorse or promote products derived from this software without 
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, 
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import math
#import thread
import numpy
import uuid
import base64
import json

#import simple_server
import msgutil
import matplotlib
#from mplh5canvas import MANAGEMENT_PORT_BASE

from matplotlib import _png
from matplotlib.backend_bases import RendererBase, GraphicsContextBase
from matplotlib.backend_bases import FigureManagerBase, FigureCanvasBase
from matplotlib.figure import Figure
from matplotlib.transforms import Affine2D
from matplotlib.path import Path
from matplotlib.colors import colorConverter, rgb2hex
from matplotlib.cbook import maxdict
from matplotlib.ft2font import FT2Font, LOAD_NO_HINTING
from matplotlib.font_manager import findfont
from matplotlib.mathtext import MathTextParser

import logging
LOGGER = logging.getLogger("mplh5canvas.backend_h5canvas")

CAP_STYLE = {'projecting' : 'square', 'butt' : 'butt', 'round': 'round',}
# mapping from matplotlib style line caps to H5 canvas

# Remember figure managers. by figure number and by IP:port
FIGURES = {}


#This command is sent to the user when a click event is fired.
#If a user clicks on a figure(not on a button) we don't want the blocker overlay to stay up until timeout
FAKE_COMMAND = "FAKE_COMMAND"
EVAL_CMD = "EVAL_JS"

def web_socket_transfer_data(request):
    """ Method to be executed with each web socket connection."""
    while True:
        canvas_manager = None
        try:
            input_line = msgutil.receive_message(request).encode('utf-8')
            message_dict = json.loads(input_line)
            LOGGER.info("Received command "+ str(message_dict))
            action = message_dict['type']
            args = message_dict['args']
            ident = message_dict['id']
            if ident not in FIGURES:
                LOGGER.debug("Not figure %s is registered to backend"%(str(ident),))
                continue
            canvas_manager = FIGURES[ident]
            if action == 'register':
                canvas_manager.register_request_handler(request)
                #canvas_manager.handle_resize(args[0], args[1])
                FIGURES[request.connection.remote_addr[0]+":"+ str(request.connection.remote_addr[1])] = canvas_manager
                continue
            method = getattr(canvas_manager, "handle_%s" % action)
            if method:
                method(**args)
            else:
                LOGGER.warning("Cannot find request method handle_%s" % action)
        except Exception, excep_:
            LOGGER.warning("Caught exception." + str(excep_))
            if canvas_manager is None:
                req_ident = (request.connection.remote_addr[0]+":"+ str(request.connection.remote_addr[1]))
                canvas_manager = FIGURES[req_ident]
                del FIGURES[req_ident]
            if canvas_manager is not None:
                LOGGER.debug("Removing registered handler")
                canvas_manager.deregister_request_handler(request)
                canvas_manager.handle_close()
            break
            

def mpl_to_css_color(color, alpha=None, is_rgb=True):
    """Convert Matplotlib color spec (or rgb tuple + alpha) into CSS string."""
    if not is_rgb:
        red, green, blue, alpha = colorConverter.to_rgba(color)
        color = (red, green, blue)
    if alpha is None and len(color) == 4:
        alpha = color[3]
    if alpha is None:
        return rgb2hex(color[:3])
    else:
        return 'rgba(%d, %d, %d, %.3g)' % (color[0] * 255, color[1] * 255, color[2] * 255, alpha)


class WebPNG(object):
    """Very simple file like object for use with the write_png method.
    Used to grab the output that would have headed to a standard file, and allow further manipulation
    such as base 64 encoding."""
    
    def __init__(self):
        self.buffer = ""
        
    def write(self, string_):
        self.buffer += string_
        
    def get_b64(self):
        return base64.b64encode(self.buffer)



class H5Frame(object):
    """
    Map the HTML frame element.
    """
    def __init__(self, frame_number=0, context_name='c'):
        self._frame_number = frame_number
        # the frame number in the current animated sequence
        self._context_name = context_name
        # the name of the context to use for drawing
        self._content = []
        # a full frame of script ready for rendering
        self._extra = []
        self._header = "frame_body_%s();" % self._context_name
        self._custom_header = False

    def _convert_obj(self, obj):
        return (isinstance(obj, unicode) and repr(obj.replace("'","`"))[1:] or 
                (isinstance(obj, float) and '%.2f' % obj or repr(obj)))

    def __getattr__(self, method_name):
        def h5_method(*args):
            """
            when frame is called in .<method_name>(<argument>) context
            """
            self._content.append('%s.%s(%s);\n' % (self._context_name, method_name,
                                               ','.join([self._convert_obj(obj) for obj in args])))
        return h5_method

    def __setattr__(self, prop, value):
        # when frame properties are assigned to .<prop> = <value>
        if prop.startswith('_'):
            self.__dict__[prop] = value
            return
        self._content.append('%s.%s=%s;\n' % (self._context_name, prop, self._convert_obj(value)))

    def moveTo(self, end_x, end_y):
        self._content.append('%s.%s(%.2f,%.2f);\n' % (self._context_name, "moveTo", end_x, end_y))

    def lineTo(self, end_x, end_y):
        self._content.append('%s.%s(%.2f,%.2f);\n' % (self._context_name, "lineTo", end_x, end_y))

    def dashedLine(self, start_x1, start_y1, end_x2, end_y2, dashes):
        """
        Draw dashed line from (x1, y1) to (x2, y2), 
        given dashes structure, and return new dash offset.
        """
        length = numpy.sqrt((end_x2 - start_x1) ** 2 + (end_y2 - start_y1) ** 2)
        if length <= 0.0:
            return dashes[0]
        dash_length = numpy.sum(dashes[1])
        # Wrap offset to fall in interval [-dash_length..0], and do one dash 
        #period extra to ensure dashed line has no gaps
        offset = -(dashes[0] % dash_length)
        num_periods = int(length // dash_length) + 2
        unit_x = (end_x2 - start_x1) / length
        unit_y = (end_y2 - start_y1) / length
        # The rest of the function can be implemented in Javascript instead, 
        #to compress the string being sent across the network
        self.moveTo(start_x1, start_y1)
        for j in xrange(num_periods):
            for i, dash_step in enumerate(dashes[1]):
                # Clip start of dash segment if it straddles (x1, y1)
                if offset < 0.0 and (offset + dash_step) > 0.0:
                    dash_step += offset
                    offset = 0.0
                # Clip end of dash segment if it straddles (x2, y2)
                if offset < length and (offset + dash_step) > length:
                    dash_step = length - offset
                # Advance to end of current dash segment
                offset += dash_step
                if offset >= 0.0 and offset <= length:
                    # Alternately draw dash and move to start of next dash
                    if i % 2 == 0:
                        self.lineTo(start_x1 + unit_x * offset, start_y1 + unit_y * offset)
                    else:
                        self.moveTo(start_x1 + unit_x * offset, start_y1 + unit_y * offset)
        return dashes[0] + (length % dash_length)

    def beginPath(self):
        self._content.append('%s.%s();\n' % (self._context_name, "beginPath"))

    def stroke(self):
        self._content.append('%s.%s();\n' % (self._context_name, "stroke"))

    def closePath(self):
        self._content.append('%s.%s();\n' % (self._context_name, "closePath"))

    def add_header(self, str_, start=False):
        if not self._custom_header:
            self._custom_header = True
            self._header = ""
        if start: 
            self._header = "%s\n" % str_ + self._header
        else: self._header += "%s\n" % str_

    def write_extra(self, str_):
        self._extra.append('%s\n' % str_)

    def write(self, str_):
        self._content.append('%s\n' % str_)

    def get_frame(self):
        return "function frame_body_%s() { %s }\n" % (self._context_name, ''.join(self._content))

    def get_frame_extra(self):
        return "function frame_body_%s() { %s\n%s }\n" % (self._context_name, ''.join(self._extra),
                                                          ''.join(self._content))

    def get_header(self):
        return "function frame_header() { %s }\n" % self._header

    def get_extra(self):
        return ''.join(self._extra)

class RendererH5Canvas(RendererBase):
    """The renderer handles drawing/rendering operations."""
    fontd = maxdict(50)

    def __init__(self, width, height, ctx, dpi=72):
        self.width = width
        self.height = height
        self.dpi = dpi
        self.ctx = ctx
        self._image_count = 0
        # used to uniquely label each image created in this figure...
        # define the JS context
        self.ctx.width = width
        self.ctx.height = height
        #self.ctx.textAlign = "center";
        self.ctx.textBaseline = "alphabetic"
        self.flip = Affine2D().scale(1, -1).translate(0, height)
        self.mathtext_parser = MathTextParser('bitmap')
        self._last_clip = None
        self._last_clip_path = None
        self._clip_count = 0

    def _set_style(self, gcontext, rgb_face=None):
        """ Set background and line width"""
        ctx = self.ctx
        if rgb_face is not None:
            alpha = gcontext.get_alpha()
            #commented the following lines because the UI widgets were not drawn correctly
#            if rgb_face[0] != 1.0 or rgb_face[1] != 1.0 or rgb_face[2] != 1.0:
#                alpha = 0.0
            ctx.fillStyle = mpl_to_css_color(rgb_face, alpha)
        ctx.strokeStyle = mpl_to_css_color(gcontext.get_rgb(), gcontext.get_alpha())
        if gcontext.get_capstyle():
            ctx.lineCap = CAP_STYLE[gcontext.get_capstyle()]
        ctx.lineWidth = self.points_to_pixels(gcontext.get_linewidth())

    def _path_to_h5(self, ctx, path, transform, clip=None, stroke=True, dashes=(None, None)):
        """Iterate over a path and produce h5 drawing directives."""
        transform = transform + self.flip
        ctx.beginPath()
        current_point = None
        dash_offset, dash_pattern = dashes
        if dash_pattern is not None:
            dash_offset = self.points_to_pixels(dash_offset)
            dash_pattern = tuple([self.points_to_pixels(dash) for dash in dash_pattern])
            
        for points, code in path.iter_segments(transform, clip=clip):
            # Shift all points by half a pixel, so that integer coordinates are
            #aligned with pixel centers instead of edges
            # This prevents lines that are one pixel wide and aligned with the 
            #pixel grid from being rendered as a two-pixel wide line
            # This happens because HTML Canvas defines (0, 0) as the *top left* 
            #of a pixel instead of the center, which causes all integer-valued 
            #coordinates to fall exactly between pixels
            points += 0.5
            if code == Path.MOVETO:
                ctx.moveTo(points[0], points[1])
                current_point = (points[0], points[1])
            elif code == Path.LINETO:
                if (dash_pattern is None) or (current_point is None):
                    ctx.lineTo(points[0], points[1])
                else:
                    dash_offset = ctx.dashedLine(current_point[0], current_point[1],
                                                 points[0], points[1], (dash_offset, dash_pattern))
                current_point = (points[0], points[1])
            elif code == Path.CURVE3:
                ctx.quadraticCurveTo(*points)
                current_point = (points[2], points[3])
            elif code == Path.CURVE4:
                ctx.bezierCurveTo(*points)
                current_point = (points[4], points[5])
            else:
                pass
        if stroke: 
            ctx.stroke()

    def _do_path_clip(self, ctx, clip):
        self._clip_count += 1
        ctx.save()
        ctx.beginPath()
        ctx.moveTo(clip[0], clip[1])
        ctx.lineTo(clip[2], clip[1])
        ctx.lineTo(clip[2], clip[3])
        ctx.lineTo(clip[0], clip[3])
        ctx.clip()

    def draw_path(self, gcontext, path, transform, rgb_face=None):
        self._set_style(gcontext, rgb_face)
        clip = self._get_gc_clip_svg(gcontext)
        clippath, cliptrans = gcontext.get_clip_path()
        ctx = self.ctx
        if clippath is not None and self._last_clip_path != clippath:
            ctx.restore()
            ctx.save()
            self._path_to_h5(ctx, clippath, cliptrans, None, stroke=False)
            ctx.clip()
            self._last_clip_path = clippath
        if self._last_clip != clip and clip is not None and clippath is None:
            ctx.restore()
            self._do_path_clip(ctx, clip)
            self._last_clip = clip
        if (clip is None and clippath is None and (self._last_clip is not None or self._last_clip_path is not None)): 
            self._reset_clip()
        if rgb_face is None and gcontext.get_hatch() is None:
            figure_clip = (0, 0, self.width, self.height)
        else:
            figure_clip = None
        self._path_to_h5(ctx, path, transform, figure_clip, dashes=gcontext.get_dashes())
        if rgb_face is not None:
            ctx.fill()
            ctx.fillStyle = '#000000'

    def _get_gc_clip_svg(self, gcontext):
        cliprect = gcontext.get_clip_rectangle()
        if cliprect is not None:
            start_x, start_y, width, height = cliprect.bounds
            start_y = self.height - (start_y + height)
            return (start_x, start_y, start_x + width, start_y + height)
        return None

    def draw_markers(self, gcontext, marker_path, marker_trans, path, trans, rgb_face=None):
        for vertices, codes in path.iter_segments(trans, simplify=False):
            if len(vertices):
                start_x, start_y = vertices[-2:]
                self._set_style(gcontext, rgb_face)
                clip = self._get_gc_clip_svg(gcontext)
                ctx = self.ctx
                self._path_to_h5(ctx, marker_path, marker_trans + Affine2D().translate(start_x, start_y), clip)
                if rgb_face is not None:
                    ctx.fill()
                    ctx.fillStyle = '#000000'

    def _slipstream_png(self, start_x, start_y, im_buffer, width, height, flip=False):
        """Insert image directly into HTML canvas as base64-encoded PNG."""
        # Shift x, y (top left corner) to the nearest CSS pixel edge, 
        #to prevent re-sampling and consequent image blurring
        start_x = math.floor(start_x + 0.5)
        start_y = math.floor(start_y + 1.5)
        # Write the image into a WebPNG object
        image_array = numpy.fromstring(im_buffer, numpy.uint8)
        image_array.shape = height, width, 4
        if flip:
            image_array = numpy.flipud(image_array)
        web_png = WebPNG()
        _png.write_png(image_array, web_png, self.dpi)
        # Extract the base64-encoded PNG and send it to the canvas
        uname = str(uuid.uuid1()).replace("-","") 
        # try to use a unique image name
        enc = "var canvas_image_%s = 'data:image/png;base64,%s';" % (uname, web_png.get_b64())
        script = "function imageLoaded_%s(ev) {\nim = ev.target; \nim_left_to_load_%s -=1;\nif (im_left_to_load_%s == 0) frame_body_%s();\n}\ncanv_im_%s = new Image();\ncanv_im_%s.onload = imageLoaded_%s;\ncanv_im_%s.src = canvas_image_%s;\n" \
                % (uname, self.ctx._context_name, self.ctx._context_name, 
                   self.ctx._context_name, uname, uname, uname, uname, uname)
        self.ctx.add_header(enc)
        self.ctx.add_header(script)
        # Once the base64 encoded image was received, draw it into the canvas
        self.ctx.write("%s.drawImage(canv_im_%s, %g, %g, %g, %g);" % (
                self.ctx._context_name, uname, start_x, start_y, width, height))
        # draw the image as loaded into canv_im_%d...
        self._image_count += 1


    def _reset_clip(self):
        self.ctx.restore()
        self._last_clip = None
        self._last_clip_path = None

    def draw_image(self, *args, **kwargs):
        """
        <1.0.0: def draw_image(self, x, y, im, bbox, 
                           clippath=None, clippath_trans=None):
        1.0.0 and up: def draw_image(self, gc, x, y, im, clippath=None):
        API for draw image changed between 0.99 and 1.0.0
        """
        start_x, start_y, imag = args[:3]
        try:
            height = imag.get_size_out()[0]
        except AttributeError:
            start_x, start_y, imag = args[1:4]
            height = imag.get_size_out()[0]
        clippath = (kwargs.has_key('clippath') and kwargs['clippath'] or None)
        if self._last_clip is not None or self._last_clip_path is not None: 
            self._reset_clip()
        if clippath is not None:
            #self._path_to_h5(self.ctx, clippath, clippath_trans, stroke=False)
            self._path_to_h5(self.ctx, clippath, clippath, stroke=False)
            self.ctx.save()
            self.ctx.clip()
        (start_x, start_y) = self.flip.transform((start_x, start_y))
        if hasattr(imag, 'flipud_out'):
            imag.flipud_out()
            flip = False
        else:
            flip = True
        rows, cols, im_buffer = imag.as_rgba_str()
        self._slipstream_png(start_x, (start_y - height), im_buffer, cols, rows, flip)
        if clippath is not None:
            self.ctx.restore()

    def _get_font(self, prop):
        """ Compute font instance, based on peoperties."""
        key = hash(prop)
        font = self.fontd.get(key)
        if font is None:
            fname = findfont(prop)
            font = self.fontd.get(fname)
            if font is None:
                font = FT2Font(str(fname))
                self.fontd[fname] = font
            self.fontd[key] = font
        font.clear()
        font.set_size(prop.get_size_in_points(), self.dpi)
        return font

    def draw_tex(self, gcontext, start_x, start_y, str_, prop, angle, ismath=False):
        LOGGER.warning("Tex support is currently not implemented. Text element '" + str(str_) + "' will not be displayed...")

    def draw_text(self, gcontext, start_x, start_y, str_val, prop, angle, ismath=False, mtext=None):
        
        if self._last_clip is not None or self._last_clip_path is not None: 
            self._reset_clip()
        if ismath:
            self._draw_mathtext(gcontext, start_x, start_y, str_val, prop, angle)
            return
        
        angle = math.radians(angle)
        descent = self.get_text_width_height_descent(str_val, prop, ismath)[2]
        start_x -= math.sin(angle) * descent
        start_y -= math.cos(angle) * descent
        ctx = self.ctx
        if angle != 0:
            ctx.save()
            ctx.translate(start_x, start_y)
            ctx.rotate(-angle)
            ctx.translate(-start_x, -start_y)
            
        font_size = self.points_to_pixels(prop.get_size_in_points())
        font_str = '%s %s %.3gpx %s, %s' % (prop.get_style(), prop.get_weight(),
                            font_size, prop.get_name(), prop.get_family()[0])
        ctx.font = font_str
        # Set the text color, draw the text and reset the color to white after
        ctx.fillStyle = mpl_to_css_color(gcontext.get_rgb(), gcontext.get_alpha())
        ctx.fillText(unicode(str_val), start_x, start_y)
        ctx.fillStyle = '#000000'
        if angle != 0:
            ctx.restore()

    def _draw_mathtext(self, gcontext, start_x, start_y, str_, prop, angle):
        """Draw math text using matplotlib.mathtext."""
        # Render math string as an image at the configured DPI, 
        #and get the image dimensions and baseline depth
        rgba = self.mathtext_parser.to_rgba(str_, color=gcontext.get_rgb(),
                    dpi=self.dpi, fontsize=prop.get_size_in_points())[0]
        height, width = rgba.shape[0], rgba.shape[1]
        angle = math.radians(angle)
        # Shift x, y (top left corner) to the nearest CSS pixel edge, 
        #to prevent resampling and consequent image blurring
        start_x = math.floor(start_x + 0.5)
        start_y = math.floor(start_y + 1.5)
        ctx = self.ctx
        if angle != 0:
            ctx.save()
            ctx.translate(start_x, start_y)
            ctx.rotate(-angle)
            ctx.translate(-start_x, -start_y)
        # Insert math text image into stream, and adjust x, y reference point 
        #to be at top left of image
        self._slipstream_png(start_x, start_y - height, rgba.tostring(), width, height)
        if angle != 0:
            ctx.restore()

    def flipy(self):
        return True

    def get_canvas_width_height(self):
        return self.width, self.height

    def get_text_width_height_descent(self, str_val, prop, ismath):
        if ismath:
            image, desc = self.mathtext_parser.parse(str_val, self.dpi, prop)
            width, height = image.get_width(), image.get_height()
        else:
            font = self._get_font(prop)
            font.set_text(str_val, 0.0, flags=LOAD_NO_HINTING)
            width, height = font.get_width_height()
            width /= 64.0  # convert from subpixels
            height /= 64.0
            desc = font.get_descent() / 64.0
        return width, height, desc

    def new_gc(self):
        return GraphicsContextH5Canvas()

    def points_to_pixels(self, points):
        # The standard desktop-publishing (Postscript) point is 1/72 of an inch
        return float(points)/72.0 * self.dpi


class GraphicsContextH5Canvas(GraphicsContextBase):
    """
    The graphics context provides the color, line styles, etc...  See the gtk
    and postscript backends for examples of mapping the graphics context
    attributes (cap styles, join styles, line widths, colors) to a particular
    backend.  In GTK this is done by wrapping a gtk.gdk.GC object and
    forwarding the appropriate calls to it using a dictionary mapping styles
    to gdk constants.  In Postscript, all the work is done by the renderer,
    mapping line styles to postscript calls.

    If it's more appropriate to do the mapping at the renderer level (as in
    the postscript backend), you don't need to override any of the GC methods.
    If it's more appropriate to wrap an instance (as in the GTK backend) and
    do the mapping here, you'll need to override several of the setter
    methods.

    The base GraphicsContext stores colors as a RGB tuple on the unit
    interval, e.g., (0.5, 0.0, 1.0). You may need to map this to colors
    appropriate for your backend.
    """
    pass



########################################################################
#
# The following functions and classes are for pylab and implement
# window/figure managers, etc...
#
########################################################################

def draw_if_interactive():
    """
    For image back-ends - is not required
    For GUI back-ends - this should be overwritten if drawing should be done in
    interactive Python mode
    """
    pass



def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    # if a main-level app must be created, this is the usual place to
    # do it -- see backend_wx, backend_wxagg and backend_tkagg for
    # examples.  Not all GUIs require explicit instantiation of a
    # main-level app (egg backend_gtk, backend_gtkagg) for pylab
    FigureClass = kwargs.pop('FigureClass', Figure)
    this_fig = FigureClass(*args, **kwargs)
    canvas = FigureCanvasH5Canvas(this_fig, num)
    FIGURES[str(num)] = canvas
    manager = FigureManagerH5Canvas(canvas, num)
    LOGGER.info("New figure created..." + str(num))
    this_fig.__dict__['show'] = canvas.draw
    # provide a show that is basically just a canvas refresh...
    return manager


class FigureCanvasH5Canvas(FigureCanvasBase):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

      figure - A Figure instance

    Note GUI templates will want to connect events for button presses,
    mouse movements and key presses to functions that call the base
    class methods button_press_event, button_release_event,
    motion_notify_event, key_press_event, and key_release_event.  See,
    e.g. backend_gtk.py, backend_wx.py and backend_tkagg.py
    """

    def __init__(self, figure, number):
        FigureCanvasBase.__init__(self, figure)
        self._user_event = None
        self._user_cmd_ret = None
        self._request_handlers = {}
        self._frame = None
        self._header = ""
        self._home_x = {}
        self._home_y = {}
        self._zoomed = False
        self._custom_content = None
        self._width, self._height = self.get_width_height()
        self.flip = Affine2D().scale(1, -1).translate(0, self._height)
        self.handle_resize(self._width, self._height)
        LOGGER.debug("Creating canvas manager for new figure %i "% number)
            

    def register_request_handler(self, request):
        """ At first connection with the client, register it, for callbacks"""
        self._request_handlers[request] = request.connection.remote_addr[0]
        # if we have a lurking frame, send it on
        if self._frame is not None:
            self.send_frame(self._header + self._frame_extra)


    def handle_user_cmd_ret(self, *args):
        """ Custom user comand"""
        if self._user_cmd_ret is not None:
            try:
                self._user_cmd_ret(*args)
            except Exception, excep_:
                LOGGER.exception("User cmd ret exception (" + str(excep_) + ")")

    def handle_user_event(self, *args):
        """ Custom user event """
        if self._user_event is not None:
            try:
                self._user_event(*args)
            except Exception, excep_:
                LOGGER.exception("User event exception (" + str(excep_) + ")")
        else: 
            LOGGER.debug("User event called but no callback registered to handle it...")

    def handle_click(self, start_x, start_y, button):
        """ 
        Currently we do not distinguish between press and release 
        on the javascript side. So call both :)
        """
        self.button_release_event(float(start_x), float(start_y), int(button))
        self.button_press_event(float(start_x), float(start_y), int(button))
        if hasattr(self.figure, 'command') and self.figure.command is not None:
            command = self.figure.command
            self.figure.command = None
            self.send_cmd(command, EVAL_CMD)
        else:
            self.send_cmd(None, FAKE_COMMAND)
            
    def handle_motion_notify(self, current_x, current_y):
        """
        Hover event for slide on canvas.
        """
        self.motion_notify_event(float(current_x), float(current_y))
        if hasattr(self.figure, 'command') and self.figure.command is not None:
            command = self.figure.command
            self.figure.command = None
            self.send_cmd(command, EVAL_CMD)
        else:
            self.send_cmd(None, FAKE_COMMAND)

    def handle_resize(self, width, height):
        """ set the figure and force a redraw..."""
        width_in = float(width) / self.figure.dpi
        height_in = float(height) / self.figure.dpi
        self.figure.set_size_inches(width_in, height_in)
        self.draw()

    def handle_close(self, *args):
        """ Pass to matlab figure.close"""
        matplotlib.pylab.close(self.figure)

    def handle_home(self, *args):
        """ Reset the plot to it's home coordinates"""
        for i in self._home_x.keys():
            self.figure.axes[i].set_xlim(self._home_x[i][0], self._home_x[i][1])
            self.figure.axes[i].set_ylim(self._home_y[i][0], self._home_y[i][1])
        self._zoomed = False
        self.draw()
        
    def handle_print_svg(self, *args):
        return self.print_svg(*args)

    def handle_zoom(self, axes, bottom_x0, bottom_y0, top_x1, top_y1):
        """ these coordinates should be the bottom left and top right of 
            the zoom bounding box, in figure pixels.."""
        axes = int(axes)
        if not self._zoomed:
            self._home_x[axes] = self.figure.axes[axes].get_xlim()
            self._home_y[axes] = self.figure.axes[axes].get_ylim()
        self._zoomed = True
        inverse = self.figure.axes[axes].transData.inverted()
        lastx, lasty = inverse.transform_point((float(bottom_x0), float(bottom_y0)))
        inverse_x, inverse_y = inverse.transform_point((float(top_x1), float(top_y1)))
        bottom_x0, bottom_y0, top_x1, top_y1 = self.figure.axes[axes].viewLim.frozen().extents

        x_min, x_max = self.figure.axes[axes].get_xlim()
        y_min, y_max = self.figure.axes[axes].get_ylim()
        twinx, twiny = False, False

        if twinx:
            bottom_x0, top_x1 = x_min, x_max
        else:
            if x_min < x_max:
                if inverse_x < lastx:  
                    bottom_x0, top_x1 = inverse_x, lastx
                else: 
                    bottom_x0, top_x1 = lastx, inverse_x
                if bottom_x0 < x_min: 
                    bottom_x0 = x_min
                if top_x1 > x_max: 
                    top_x1 = x_max
            else:
                if inverse_x > lastx:  
                    bottom_x0, top_x1 = inverse_x, lastx
                else: 
                    bottom_x0, top_x1 = lastx, inverse_x
                if bottom_x0 > x_min: 
                    bottom_x0 = x_min
                if top_x1 < x_max: 
                    top_x1 = x_max

        if twiny:
            bottom_y0, top_y1 = y_min, y_max
        else:
            if y_min < y_max:
                if inverse_y < lasty:  
                    bottom_y0, top_y1 = inverse_y, lasty
                else: 
                    bottom_y0, top_y1 = lasty, inverse_y
                if bottom_y0 < y_min: 
                    bottom_y0 = y_min
                if top_y1 > y_max: 
                    top_y1 = y_max
            else:
                if inverse_y > lasty:  
                    bottom_y0, top_y1 = inverse_y, lasty
                else: 
                    bottom_y0, top_y1 = lasty, inverse_y
                if bottom_y0 > y_min: 
                    bottom_y0 = y_min
                if top_y1 < y_max: 
                    top_y1 = y_max
        self.figure.axes[axes].set_xlim((bottom_x0, top_x1))
        self.figure.axes[axes].set_ylim((bottom_y0, top_y1))
        self.draw()

    def deregister_request_handler(self, request):
        """ Remove back-reference from server"""
        del self._request_handlers[request]


    def draw(self, ctx_override='c', *args, **kwargs):
        """
        Draw the figure using the renderer
        """
        width, height = self.get_width_height()
        ctx = H5Frame(context_name=ctx_override)
        # the context to write the js in...
        renderer = RendererH5Canvas(width, height, ctx, dpi=self.figure.dpi)
        ctx.write_extra("resize_canvas(id," + str(width) + "," + 
                        str(height) + ");")
        ctx.write_extra("native_w[id] = " + str(width) + ";")
        ctx.write_extra("native_h[id] = " + str(height) + ";")
        # clear the canvas...  
        self.figure.draw(renderer)
        for i, axe in enumerate(self.figure.axes):
            corners = axe.bbox.corners()
            bb_str = ""
            for corner in corners: 
                bb_str += str(corner[0]) + "," + str(corner[1]) + ","
            ctx.add_header("ax_bb[" + str(i) + "] = [" + bb_str[:-1] + "];")
        if renderer._image_count > 0:
            ctx.add_header("var im_left_to_load_%s = %i;" % (ctx._context_name, 
                                        renderer._image_count), start=True)
        else:
            ctx.add_header("frame_body_%s();" % ctx._context_name)
            # if no image we can draw the frame body immediately..
        self._header = ctx.get_header()
        self._frame = ctx.get_frame()
        self._frame_extra = ctx.get_frame_extra()
        # additional script commands needed for other functions than drawing
        self._width, self._height = self.get_width_height()
        # redo my height and width...
        self.send_frame(self._header + self._frame_extra)

    def send_cmd(self, cmd=None, cmd_type="EVAL_CMD"):
        """Send a string of javascript to be executed on the client 
        side of each connected user."""
        command_dict = {"exec_user_cmd" : cmd_type, 'parameters' : cmd}
        self.send_frame(json.dumps(command_dict))

    def send_frame(self, frame):
        """ Back communication towards the client."""
        for req_ref in self._request_handlers.keys():
            try:
                msgutil.send_message(req_ref, frame.decode('utf-8'))
            except AttributeError:
                # connection has gone
                LOGGER.debug("Connection" + str(req_ref.connection.remote_addr[0])  + "has gone. Closing...")
            except Exception, excep_:
                LOGGER.exception( "Failed to send message %s", excep_)

    def show(self):
        """Not implemented in this function"""
        LOGGER.warning("Show called... Not implemented in this function...")

    filetypes = {'js': 'HTML5 Canvas'}

    def print_js(self, filename, *args, **kwargs):
        LOGGER.debug("Print js called with args" + str(args) + "and **kwargs" + str(kwargs))
        width, height = self.get_width_height()
        writer = open(filename, 'w')
        renderer = RendererH5Canvas(width, height, writer, dpi=self.figure.dpi)
        self.figure.draw(renderer)

    def get_default_filetype(self):
        return 'js'


class FigureManagerH5Canvas(FigureManagerBase):
    """
    Wrap everything up into a window for the pylab interface.
    For non interactive backends, the base class does all the work.
    """
    
    def __init__(self, canvas, num):
        self.canvas = canvas
        FigureManagerBase.__init__(self, canvas, num)

    def destroy(self, *args):
        LOGGER.debug("Destroy called on figure manager " + str(args))
        self.canvas.handle_close()

    def show(self):
        LOGGER.debug("Show called for figure manager")




