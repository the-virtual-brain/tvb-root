#!/bin/bash
# Runs the tests for TVB project.

echo 'Executing clean before tests...'
if [ "$1" ]; then
    # Make sure TVB application is not running....
    python app.py stop

	# Run tests using PostgreSQL DB
	python app.py clean TEST_POSTGRES_PROFILE
	
	echo 'Starting TVB tests on PostgreSQL DB ...'
	python -m tvb.tests.framework.main_tests xml -profile TEST_POSTGRES_PROFILE
	
else
    # Make sure TVB application is not running....
    python app.py stop

	# Run tests using SQLite as DB
	python app.py clean TEST_SQLITE_PROFILE
	
	echo 'Starting TVB tests on SQLite DB ...'
	python -m tvb.tests.framework.main_tests xml -profile TEST_SQLITE_PROFILE
fi

echo 'Starting TVB Scientific Library tests'
python -m tvb.tests.library.main_tests xml

echo 'Tests done.'

exit 0

