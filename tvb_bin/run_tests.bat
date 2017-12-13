@echo off
rem Runs the tests for TVB project.

rem Make sure TVB application is not running....
python tvb_bin\app.py stop

echo 'Executing clean before tests...'
python tvb_bin\app.py clean TEST_SQLITE_PROFILE

echo 'Starting TVB tests...'
pytest --pyargs tvb.tests.framework --profile=TEST_SQLITE_PROFILE --junitxml=TEST_OUTPUT/TEST-RESULTS.xml

echo 'Starting TVB Scientific Library tests'
pytest --pyargs tvb.tests.library --junitxml=TEST_OUTPUT/TEST-LIBRARY-RESULTS.xml

echo 'Tests done.'

exit 0

