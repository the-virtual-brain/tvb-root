REM Package Geodesic Distance for Pypi
rmdir dist
mkdir dist

echo "====================================="
echo "Packing: externals/tvb_gdist"
echo "====================================="

cd ..\externals\tvb_gdist
python setup.py sdist
python setup.py bdist_wheel

move dist\* ..\..\dist\
rmdir dist
rmdir build
cd ..\..