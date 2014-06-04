rm bench_nsim.txt
rm bench_speed.txt
total=512
for i in `seq 32 32 $total`
do
	echo $i
	python demos/benchmark_device.py $i 30.0 >> bench_nsim.txt
	#v=`echo "e($i/$total*3*l(10))" | bc -l`
	#echo $v
#	python demos/benchmark_device.py 32 $v >> bench_speed.txt
done


	
