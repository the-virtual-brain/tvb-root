"""
XML Test Runner for PyUnit
"""

# Written by Sebastian Rittau <srittau@jroger.in-berlin.de> and placed in
# the Public Domain. With contributions by Paolo Borelli and others.
# Updated by Lia Domide <lia.domide@codemart.ro> by adding report of skipped tests.

from __future__ import with_statement

__version__ = "0.1"

import sys
import time
import traceback
import unittest
from xml.sax.saxutils import escape
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO



class _TestInfo(object):
    """
    Information about a particular test.
    Used by _XMLTestResult.
    """

    def __init__(self, test, time):
        (self._class, self._method) = test.id().rsplit(".", 1)
        self._time = time
        self._error = None
        self._failure = None
        self._skipped = None

    @staticmethod
    def create_success(test, time):
        """Create a _TestInfo instance for a successful test."""
        return _TestInfo(test, time)

    @staticmethod
    def create_failure(test, time, failure):
        """Create a _TestInfo instance for a failed test."""
        info = _TestInfo(test, time)
        info._failure = failure
        return info

    @staticmethod
    def create_error(test, time, error):
        """Create a _TestInfo instance for an erroneous test."""
        info = _TestInfo(test, time)
        info._error = error
        return info

    @staticmethod
    def create_skipped(test, time, skipped):
        """Create a _TestInfo instance for a skipped test."""
        info = _TestInfo(test, time)
        info._skipped = skipped
        return info

    def print_report(self, stream):
        """ Print information about this test case in XML format to the supplied stream. """
        stream.write('  <testcase classname="%(class)s" name="%(method)s" time="%(time).4f">' %
                     {"class": self._class,
                      "method": self._method,
                      "time": self._time
                      })
        if self._failure is not None:
            self._print_error(stream, 'failure', self._failure)
        if self._error is not None:
            self._print_error(stream, 'error', self._error)
        if self._skipped is not None:
            self._print_skipped(stream, self._skipped)
        stream.write('</testcase>\n')

    def _print_error(self, stream, tagname, error):
        """Print information from a failure or error to the supplied stream."""
        text = escape(str(error[1]))
        stream.write('\n')
        stream.write('    <%s type="%s">%s\n' % (tagname, _clsname(error[0]), text))
        tb_stream = StringIO()
        traceback.print_tb(error[2], None, tb_stream)
        stream.write(escape(tb_stream.getvalue()))
        stream.write('    </%s>\n' % tagname)
        stream.write('  ')
        
    def _print_skipped(self, stream, skip_reason):
        """ Print the skip reason"""
        stream.write('\n')
        stream.write('    <skipped reason="%s"/>\n' % str(skip_reason))
        stream.write('  ')



def _clsname(cls):
    return cls.__module__ + "." + cls.__name__



class _XMLTestResult(unittest.TestResult):
    """
    A test result class that stores result as XML.
    Used by XMLTestRunner.
    """

    def __init__(self, classname):
        unittest.TestResult.__init__(self)
        self._test_name = classname
        self._start_time = None
        self._tests = []
        self._error = None
        self._failure = None
        self._skipped = None

    def startTest(self, test):
        unittest.TestResult.startTest(self, test)
        self._error = None
        self._failure = None
        self._skipped = None
        self._start_time = time.time()

    def stopTest(self, test):
        time_taken = time.time() - self._start_time
        unittest.TestResult.stopTest(self, test)
        if self._error:
            info = _TestInfo.create_error(test, time_taken, self._error)
        elif self._failure:
            info = _TestInfo.create_failure(test, time_taken, self._failure)
        elif self.skipped:
            info = _TestInfo.create_skipped(test, time_taken, self._skipped)
        else:
            info = _TestInfo.create_success(test, time_taken)
        self._tests.append(info)

    def addError(self, test, err):
        unittest.TestResult.addError(self, test, err)
        self._error = err

    def addFailure(self, test, err):
        unittest.TestResult.addFailure(self, test, err)
        self._failure = err

    def addSkip(self, test, reason):
        unittest.TestResult.addSkip(self, test, reason)
        self._skipped = reason

    def print_report(self, stream, logs_stream, time_taken, out, err):
        """
        Prints the XML report to the supplied stream.
        The time the tests took to perform as well as the captured standard
        output and standard error streams must be passed in.
        """
        stream.write('<testsuite errors="%(e)d" failures="%(f)d" skipped="%(s)d" ' % {"e": len(self.errors),
                                                                                      "f": len(self.failures),
                                                                                      "s": len(self.skipped)})
        stream.write('name="%(n)s" tests="%(t)d" time="%(time).3f">\n' %
                     {"n": self._test_name,
                      "t": self.testsRun,
                      "time": time_taken,
                      })
        for info in self._tests:
            info.print_report(stream)

        stream.write('</testsuite>\n')

        logs_stream.write("OUTPUT: \n \n\n")
        logs_stream.write(out)
        logs_stream.write("\n\n\n ERRORS: \n \n\n")
        logs_stream.write(err)



class XMLTestRunner(object):
    """
    A test runner that stores results in XML format compatible with JUnit.

    XMLTestRunner(stream=None) -> XML test runner

    The XML file is written to the supplied stream. If stream is None, the
    results are stored in a file called TEST-<module>.<class>.xml in the
    current working directory (if not overridden with the path property),
    where <module> and <class> are the module and class name of the test class.
    """

    def __init__(self, stream, logs_stream):
        self._stream = stream
        self._logs_stream = logs_stream
        self._path = "."


    def run(self, test):
        """Run the given test case or test suite."""
        class_ = test.__class__
        classname = class_.__module__ + "." + class_.__name__

        result = _XMLTestResult(classname)
        start_time = time.time()

        with _fake_std_streams():
            test(result)
            try:
                out_s = sys.stdout.getvalue()
            except AttributeError:
                out_s = ""
            try:
                err_s = sys.stderr.getvalue()
            except AttributeError:
                err_s = ""

        time_taken = time.time() - start_time
        result.print_report(self._stream, self._logs_stream, time_taken, out_s, err_s)

        return result

    def _set_path(self, path):
        self._path = path

    path = property(lambda self: self._path, _set_path, None, """The path where the XML files are stored.
    This property is ignored when the XML file is written to a file stream.""")



class _fake_std_streams(object):

    def __enter__(self):
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr



if __name__ == "__main__":
    unittest.main()
