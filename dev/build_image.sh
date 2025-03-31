rt=$1

case $rt in
    cpu)
	cp dxgb_bench/dev/Dockerfile.cpu Dockerfile
	;;
    gpu)
	cp dxgb_bench/dev/Dockerfile.gpu Dockerfile
	;;
    *)
	echo "Invalid option. Expected [cpu|gpu]."
	exit -1
	;;
esac

docker build . -t dxgb-bench-$rt

rm Dockerfile
