
cd ..

sh stop.sh
rm -R ~/TVB

cd snapshot

tar -zxf TVB_snapshot.tar.gz -C ~/

cd ..

sh start_local.sh
