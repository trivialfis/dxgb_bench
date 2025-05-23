#!/usr/bin/env bash

set -euox pipefail

DEB_ARM="nsight-systems-2025.3.1_2025.3.1.90-1_arm64.deb"
DEB_X86="nsight-systems-2025.3.1_2025.3.1.90-1_amd64.deb"
URL_ARM="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/${DEB_ARM}"
URL_X86="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/${DEB_X86}"

arch=$1

case $arch in
    aarch)
	wget $URL_ARM
	echo "02b078c20d0aad765f2695fdbcc33ba3d2152fae9d0c994ae8ea3ce9a9278c5b  ./${DEB_ARM}" | shasum -a 256 --check
	apt install ./${DEB_ARM} -y
	rm ./${DEB_ARM}
	;;
    x86)
	wget $URL_X86
	echo "43b9b97a050ac6cfbd2dc70df60eab6809d5055b222971638a555c9d9da8a1c9  ./${DEB_X86}" | shasum -a 256 --check
	apt install ./${DEB_X86} -y
	rm ./${DEB_X86}
	;;
    *)
	echo "Invalid option. Expected [aarch|x86]."
	exit -1
	;;
esac
