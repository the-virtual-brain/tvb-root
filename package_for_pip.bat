REM Package Geodesic Distance for Pypi
rmdir dist
mkdir dist

echo "====================================="
echo "Packing: externals/geodesic_distance"
echo "====================================="

cd externals\geodesic_distance
python setup.py sdist
python setup.py bdist_wheel

move dist\* ..\..\dist\
rmdir dist
rmdir build
cd ..\..