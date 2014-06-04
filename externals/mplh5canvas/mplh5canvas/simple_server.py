# Modifications 2011 by Simon Ratcliffe (sratcliffe@ska.ac.za)
#
# Based on standalone websocket server provided as part of the 
# mod_pywebsocket package
# 
# Copyright 2011, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import BaseHTTPServer
import CGIHTTPServer
import SocketServer
import os
import select
import socket
import threading
import memorizingfile
import logging

from mod_pywebsocket import dispatch
from mod_pywebsocket import handshake
from mod_pywebsocket import http_header_util
from mod_pywebsocket import util

logger = logging.getLogger("mplh5canvas.simple_server")



class _StandaloneConnection(object):
    """Mimic mod_python mp_conn."""

    def __init__(self, request_handler):
        """Construct an instance.

        Args:
            request_handler: A WebSocketRequestHandler instance.
        """

        self._request_handler = request_handler

    def get_local_addr(self):
        """Getter to mimic mp_conn.local_addr."""

        return (self._request_handler.server.server_name,
                self._request_handler.server.server_port)
    local_addr = property(get_local_addr)

    def get_remote_addr(self):
        """Getter to mimic mp_conn.remote_addr.

        Setting the property in __init__ won't work because the request
        handler is not initialized yet there."""

        return self._request_handler.client_address
    remote_addr = property(get_remote_addr)

    def write(self, data):
        """Mimic mp_conn.write()."""

        return self._request_handler.wfile.write(data)

    def read(self, length):
        """Mimic mp_conn.read()."""

        return self._request_handler.rfile.read(length)

    def get_memorized_lines(self):
        """Get memorized lines."""

        return self._request_handler.rfile.get_memorized_lines()


class _StandaloneRequest(object):
    """Mimic mod_python request."""

    def __init__(self, request_handler, use_tls):
        """Construct an instance.

        Args:
            request_handler: A WebSocketRequestHandler instance.
        """

        self._logger = util.get_class_logger(self)

        self._request_handler = request_handler
        self.connection = _StandaloneConnection(request_handler)
        self.protocol = 'HTTP/1.1'
        self._use_tls = use_tls

    def get_uri(self):
        """Getter to mimic request.uri."""

        return self._request_handler.path
    uri = property(get_uri)

    def get_method(self):
        """Getter to mimic request.method."""

        return self._request_handler.command
    method = property(get_method)

    def get_headers_in(self):
        """Getter to mimic request.headers_in."""

        return self._request_handler.headers
    headers_in = property(get_headers_in)

    def is_https(self):
        """Mimic request.is_https()."""

        return self._use_tls

    def _drain_received_data(self):
        """Don't use this method from WebSocket handler. Drains unread data
        in the receive buffer.
        """

        raw_socket = self._request_handler.connection
        drained_data = util.drain_received_data(raw_socket)

        if drained_data:
            self._logger.debug(
                'Drained data following close frame: %r', drained_data)



 # dummy to hold minimally required options
class Options(object):
    pass



class WebSocketServer(SocketServer.ThreadingMixIn, BaseHTTPServer.HTTPServer):
    """HTTPServer specialized for WebSocket."""

    # Overrides SocketServer.ThreadingMixIn.daemon_threads
    daemon_threads = True
    # Overrides BaseHTTPServer.HTTPServer.allow_reuse_address
    allow_reuse_address = True
    STATUS_NORMAL = 1000

    def extra_handshake(self, request):
        logger.debug("Doing extra initial handshake...")

    def web_socket_transfer_data(self, request):
        while True:
            line = request.ws_stream.receive_message()
            if line is None:
                return
            if isinstance(line, unicode):
                logger.info("Echo test: %s" % line)
                request.ws_stream.send_message(line, binary=False)
                if line == u'end':
                    return
            else:
                request.ws_stream.send_message(line, binary=True)

    def closing_handshake(self, request):
        logger.debug("Issued default closing handshake...")
        return self.STATUS_NORMAL, ''


    def __init__(self, server_address, transfer_data, RequestHandlerClass):
        """Override SocketServer.TCPServer.__init__ to set SSL enabled
        socket object to self.socket before server_bind and server_activate,
        if necessary.
        """
        options = Options()
        options.use_tls = False
        options.cgi_directories = []
        options.is_executable_method = None
        options.dispatcher = dispatch.Dispatcher('.', None)
        options.dispatcher._handler_suite_map['/echo'] = dispatch._HandlerSuite(self.extra_handshake,
                                                                                self.web_socket_transfer_data,
                                                                                self.closing_handshake)
         # add an echo handler for testing purposes
        options.dispatcher._handler_suite_map['/'] = dispatch._HandlerSuite(self.extra_handshake, transfer_data,
                                                                            self.closing_handshake)
         # add the supplied transfer method as the default handler
        options.allow_draft75 = True
        options.strict = False
        self.request_queue_size = 128
        self.__ws_is_shut_down = threading.Event()
        self.__ws_serving = False

        SocketServer.BaseServer.__init__(
            self, server_address, RequestHandlerClass)

        # Expose the options object to allow handler objects access it. We name
        # it with websocket_ prefix to avoid conflict.
        self.websocket_server_options = options

        self._create_sockets()
        self.server_bind()
        self.server_activate()

    def _create_sockets(self):
        self.server_name, self.server_port = self.server_address
        self._sockets = []
        if not self.server_name:
            addrinfo_array = [
                (self.address_family, self.socket_type, '', '', '')]
        else:
            addrinfo_array = socket.getaddrinfo(self.server_name,
                                                self.server_port,
                                                socket.AF_UNSPEC,
                                                socket.SOCK_STREAM,
                                                socket.IPPROTO_TCP)
        for addrinfo in addrinfo_array:
            logger.debug('Create socket on: %r', addrinfo)
            family, socktype, proto, canonname, sockaddr = addrinfo
            try:
                socket_ = socket.socket(family, socktype)
            except Exception, e:
                logger.warning('Skip by failure: %r', e)
                continue

            if self.websocket_server_options.use_tls:
                import OpenSSL
                ctx = OpenSSL.SSL.Context(OpenSSL.SSL.SSLv23_METHOD)
                ctx.use_privatekey_file(self.websocket_server_options.private_key)
                ctx.use_certificate_file(self.websocket_server_options.certificate)
                socket_ = OpenSSL.SSL.Connection(ctx, socket_)
            self._sockets.append((socket_, addrinfo))


    def server_bind(self):
        """Override SocketServer.TCPServer.server_bind to enable multiple sockets bind.
        """
        for socket_, addrinfo in self._sockets:
            logger.debug('Bind on: %r', addrinfo)
            if self.allow_reuse_address:
                socket_.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            socket_.bind(self.server_address)

    def server_activate(self):
        """Override SocketServer.TCPServer.server_activate to enable multiple
        sockets listen.
        """

        failed_sockets = []

        for socketinfo in self._sockets:
            socket_, addrinfo = socketinfo
            logger.debug('Listen on: %r', addrinfo)
            try:
                socket_.listen(self.request_queue_size)
            except Exception, e:
                logger.warning('Skip by failure: %r', e)
                socket_.close()
                failed_sockets.append(socketinfo)

        for socketinfo in failed_sockets:
            self._sockets.remove(socketinfo)

    def server_close(self):
        """Override SocketServer.TCPServer.server_close to enable multiple
        sockets close.
        """
        for socketinfo in self._sockets:
            socket_, addrinfo = socketinfo
            logger.debug('Close on: %r', addrinfo)
            socket_.close()

    def fileno(self):
        """Override SocketServer.TCPServer.fileno."""

        logger.error('Not supported: fileno')
        return self._sockets[0][0].fileno()

    def handle_error(self, rquest, client_address):
        """Override SocketServer.handle_error."""

        logger.error(
            ('Exception in processing request from: %r' % (client_address,)) +
            '\n' + util.get_stack_trace())
        # Note: client_address is a tuple. To match it against %r, we need the
        # trailing comma.

    def serve_forever(self, poll_interval=0.5):
        """Override SocketServer.BaseServer.serve_forever."""

        self.__ws_serving = True
        self.__ws_is_shut_down.clear()
        handle_request = self.handle_request
        if hasattr(self, '_handle_request_noblock'):
            handle_request = self._handle_request_noblock
        else:
            logger.warning('fallback to blocking request handler')
        try:
            while self.__ws_serving:
                r, w, e = select.select(
                    [socket_[0] for socket_ in self._sockets],
                    [], [], poll_interval)
                for socket_ in r:
                    self.socket = socket_
                    handle_request()
                self.socket = None
        finally:
            self.__ws_is_shut_down.set()

    def shutdown(self):
        """Override SocketServer.BaseServer.shutdown."""

        self.__ws_serving = False
        self.__ws_is_shut_down.wait()


class WebSocketRequestHandler(CGIHTTPServer.CGIHTTPRequestHandler):
    """CGIHTTPRequestHandler specialized for WebSocket."""
    _transfer_data = None

    def setup(self):
        """Override SocketServer.StreamRequestHandler.setup to wrap rfile
        with MemorizingFile.

        This method will be called by BaseRequestHandler's constructor
        before calling BaseHTTPRequestHandler.handle.
        BaseHTTPRequestHandler.handle will call
        BaseHTTPRequestHandler.handle_one_request and it will call
        WebSocketRequestHandler.parse_request.
        """

        # Call superclass's setup to prepare rfile, wfile, etc. See setup
        # definition on the root class SocketServer.StreamRequestHandler to
        # understand what this does.
        CGIHTTPServer.CGIHTTPRequestHandler.setup(self)

        self.rfile = memorizingfile.MemorizingFile(
            self.rfile,
            max_memorized_lines=1024)

    def __init__(self, request, client_address, server):
        self._options = server.websocket_server_options

        # Overrides CGIHTTPServerRequestHandler.cgi_directories.
        self.cgi_directories = self._options.cgi_directories
        # Replace CGIHTTPRequestHandler.is_executable method.
        if self._options.is_executable_method is not None:
            self.is_executable = self._options.is_executable_method

        self._request = _StandaloneRequest(self, self._options.use_tls)

        # This actually calls BaseRequestHandler.__init__.
        CGIHTTPServer.CGIHTTPRequestHandler.__init__(
            self, request, client_address, server)

    def parse_request(self):
        """Override BaseHTTPServer.BaseHTTPRequestHandler.parse_request.

        Return True to continue processing for HTTP(S), False otherwise.

        See BaseHTTPRequestHandler.handle_one_request method which calls
        this method to understand how the return value will be handled.
        """

        # We hook parse_request method, but also call the original
        # CGIHTTPRequestHandler.parse_request since when we return False,
        # CGIHTTPRequestHandler.handle_one_request continues processing and
        # it needs variables set by CGIHTTPRequestHandler.parse_request.
        #
        # Variables set by this method will be also used by WebSocket request
        # handling. See _StandaloneRequest.get_request, etc.
        cgi_req = CGIHTTPServer.CGIHTTPRequestHandler.parse_request(self)
        if not cgi_req:
            return False
        host, port, resource = http_header_util.parse_uri(self.path)
        if resource is None:
            logger.warning('invalid uri %r' % self.path)
            return True
        server_options = self.server.websocket_server_options
        if host is not None:
            validation_host = server_options.validation_host
            if validation_host is not None and host != validation_host:
                logger.warning('invalid host %r (expected: %r)' % (host, validation_host))
                return True
        if port is not None:
            validation_port = server_options.validation_port
            if validation_port is not None and port != validation_port:
                logger.warning('invalid port %r (expected: %r)' % (port, validation_port))
                return True
        self.path = resource

        try:
            # Fallback to default http handler for request paths for which
            # we don't have request handlers.
            if not self._options.dispatcher.get_handler_suite(self.path):
                self.path = "/"
                # we fall back to our default method if nothing higher is around...
            try:
                handshake.do_handshake(
                    self._request,
                    self._options.dispatcher,
                    allowDraft75=self._options.allow_draft75,
                    strict=self._options.strict)
            except handshake.AbortedByUserException, e:
                logger.warning('%s' % e)
                return False
            try:
                self._request._dispatcher = self._options.dispatcher
                self._options.dispatcher.transfer_data(self._request)
            except dispatch.DispatchException, e:
                logger.warning('%s' % e)
                return False
            except handshake.AbortedByUserException, e:
                logger.warning('%s' % e)
            except Exception:
                # Catch exception in transfer_data.
                # In this case, handshake has been successful, so just log
                # the exception and return False. User has closed browser.
                #logger.warning('%s' % e)
                #logger.warning('%s' % util.get_stack_trace())
                pass
        except dispatch.DispatchException, e:
            logger.warning('%s' % e)
            self.send_error(e.status)
        except handshake.HandshakeException, e:
            # Handshake for ws(s) failed. Assume http(s).
            logger.info('%s' % e)
            self.send_error(e.status)
        except Exception, e:
            logger.warning('%s' % e)
            logger.warning('%s' % util.get_stack_trace())
        return False

    def log_request(self, code='-', size='-'):
        """Override BaseHTTPServer.log_request."""

        logger.info('"%s" %s %s', self.requestline, str(code), str(size))

    def log_error(self, *args):
        """Override BaseHTTPServer.log_error."""

        # Despite the name, this method is for warnings than for errors.
        # For example, HTTP status code is logged by this method.
        logger.warning('%s - %s' % (self.address_string(), (args[0] % args[1:])))

    def is_cgi(self):
        """Test whether self.path corresponds to a CGI script.

        Add extra check that self.path doesn't contains ..
        Also check if the file is a executable file or not.
        If the file is not executable, it is handled as static file or dir
        rather than a CGI script.
        """

        if CGIHTTPServer.CGIHTTPRequestHandler.is_cgi(self):
            if '..' in self.path:
                return False
            # strip query parameter from request path
            resource_name = self.path.split('?', 2)[0]
            # convert resource_name into real path name in filesystem.
            scriptfile = self.translate_path(resource_name)
            if not os.path.isfile(scriptfile):
                return False
            if not self.is_executable(scriptfile):
                return False
            return True
        return False

