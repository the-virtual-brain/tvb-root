cd ../ui_test
rem export MAVEN_OPTS='-Xmx1024m -XX:MaxPermSize=256m'
call mvn -PsimpleLoadTestOnly clean install -Dmaven.test.skip=true