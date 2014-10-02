@echo off
rem Runs the tests for TVB project.

rem Make sure TVB application is not running....
python app.py stop

echo 'Executing clean before tests...'
python app.py clean TEST_SQLITE_PROFILE

echo 'Starting TVB tests...'
python -m tvb.tests.framework.main_tests TEST_SQLITE_PROFILE xml

echo 'Starting TVB Scientific Library tests'
python -m tvb.tests.library.main_tests xml

echo 'Tests done.'

exit 0

