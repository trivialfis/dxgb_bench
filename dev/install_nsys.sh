#!/usr/bin/env bash

set -euox pipefail

DEB_ARM="nsight-systems-2026.3.1_2026.3.1.157-1_arm64.deb"
DEB_X86="nsight-systems-2026.3.1_2026.3.1.157-1_amd64.deb"
URL_ARM="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_3/${DEB_ARM}"
URL_X86="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_3/${DEB_X86}"

arch=$1

case $arch in
    aarch)
	wget $URL_ARM
	echo "a4b202203b79525a7f24fb3b0dd1f6a52766d72979e8371db0e40b51962d6694  ./${DEB_ARM}" | shasum -a 256 --check
	apt install ./${DEB_ARM} -y
	rm ./${DEB_ARM}
	;;
    x86)
	wget $URL_X86
	echo "5ee19712bab10f3f1848493ffe808d1bf540b5c6bdf0e06ac9da867dab28935b  ./${DEB_X86}" | shasum -a 256 --check
	apt install ./${DEB_X86} -y
	rm ./${DEB_X86}
	;;
    *)
	echo "Invalid option. Expected [aarch|x86]."
	exit -1
	;;
esac
