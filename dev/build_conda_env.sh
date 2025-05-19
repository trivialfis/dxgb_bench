#!/usr/bin/env bash

set -euox pipefail

ARCH=$1

case $ARCH in
    aarch)
	echo "Build aarch64 conda environment."
	mamba env update -n base -f ws/xgboost_aarch_dev.yml -v && mamba clean --all --yes
	;;
    x86)
	echo "Build x86 conda environment."
	mamba env update -n base -f ws/xgboost_dev.yml -v && mamba clean --all --yes
	;;
    *)
	echo "Invalid option. Expected [aarch|x86], got ${ARCH}"
	exit -1
	;;
esac
