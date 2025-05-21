Build an image:

``` sh
	python ./dxgb_bench/dev/build_image.py --target=gpu --arch=x86 --sm=89
```

``` sh
	python ./dxgb_bench/dev/build_image.py --target=gpu --arch=aarch --sm=90a
```

Build nvbandwidth:
``` sh
    docker build . -f ./Dockerfile.nvbandwidth -t nvbandwidth:latest
```
