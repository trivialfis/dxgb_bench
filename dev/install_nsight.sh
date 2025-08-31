#!/usr/bin/env bash

set -euox pipefail

DEB_ARM="nsight-systems-2025.3.1_2025.3.1.90-1_arm64.deb"
DEB_X86="nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb"
URL_ARM="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/${DEB_ARM}"
URL_X86="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_5/${DEB_X86}"

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
	echo "b49be4830a9f550ce2dcd7412a5b93527014e3a57014d90f4b37e6ba65909cbc  ./${DEB_X86}" | shasum -a 256 --check
	apt install ./${DEB_X86} -y
	rm ./${DEB_X86}
	;;
    *)
	echo "Invalid option. Expected [aarch|x86]."
	exit -1
	;;
esac
