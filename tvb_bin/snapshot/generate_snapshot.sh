
rm ~/TVB_snapshot.tar.gz

rm ./TVB_snapshot.tar.gz

tar -zcf TVB_snapshot.tar.gz -X excludes.txt -C ~/ TVB
