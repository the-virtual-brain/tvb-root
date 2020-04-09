#!/bin/bash
# Runs the tests for TVB project.

echo 'Executing clean before tests...'
rm -rf TEST_OUTPUT

if [[ "$1" ]]; then

    COUNT=0
    until pg_isready -U postgres || [[ ${COUNT} -eq 5 ]]; do
        echo "Postgres is unavailable - sleeping"
        sleep 1
        let COUNT=COUNT+1
    done

    # Make sure TVB application is not running....
    python tvb_bin/app.py stop TEST_POSTGRES_PROFILE

	# Run tests using PostgreSQL DB
	python tvb_bin/app.py clean TEST_POSTGRES_PROFILE
	
	echo 'Starting TVB tests on PostgreSQL DB ...'
	mkdir TEST_OUTPUT
	pytest ../framework_tvb/tvb/tests/framework --profile=TEST_POSTGRES_PROFILE --junitxml=TEST_OUTPUT/results_frw.xml > TEST_OUTPUT/frw.out 2>&1
	
else
    # Make sure TVB application is not running....
    python tvb_bin/app.py stop TEST_SQLITE_PROFILE

	# Run tests using SQLite as DB
	python tvb_bin/app.py clean TEST_SQLITE_PROFILE
	
	echo 'Starting TVB tests on SQLite DB ...'
	mkdir TEST_OUTPUT
	pytest --pyargs tvb.tests.framework --junitxml=TEST_OUTPUT/results_frw.xml > TEST_OUTPUT/frw.out 2>&1

fi

echo 'Starting TVB Scientific Library tests'
pytest --pyargs tvb.tests.library --junitxml=TEST_OUTPUT/results_lib.xml > TEST_OUTPUT/lib.out 2>&1

echo 'Tests done.'

exit 0

