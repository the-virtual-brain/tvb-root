@echo off
rem Runs the tests for TVB project.

rem Make sure TVB application is not running....
python tvb_bin\app.py stop

echo 'Executing clean before tests...'
python tvb_bin\app.py clean TEST_SQLITE_PROFILE
mkdir TEST_OUTPUT

echo 'Starting TVB tests...'
pytest --pyargs tvb.tests.framework --junitxml=TEST_OUTPUT\TEST-RESULTS.xml > TEST_OUTPUT\TEST.out 2>&1

echo 'Starting TVB Scientific Library tests'
pytest --pyargs tvb.tests.library --junitxml=TEST_OUTPUT\TEST-LIBRARY-RESULTS.xml > TEST_OUTPUT\TEST-LIBRARY.out 2>&1

echo 'Tests done.'

exit 0

