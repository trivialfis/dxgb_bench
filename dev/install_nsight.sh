#!/usr/bin/env bash

set -euox pipefail

URL_ARM="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_2/nsight-systems-cli-2025.2.1_2025.2.1.130-1_arm64.deb"
URL_X86="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_2/NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb"

arch=$1

case $arch in
    aarch)
	wget $URL_ARM
	echo "a1d3e95fd8f1c52791d0b9f97a99d09fdb41a2d1d58db8dc8b493e1ad90278e9  ./nsight-systems-cli-2025.2.1_2025.2.1.130-1_arm64.deb"  | shasum -a 256 --check
	apt install ./nsight-systems-cli-2025.2.1_2025.2.1.130-1_arm64.deb -y
	;;
    x86)
	wget $URL_X86
	echo "e36f4d1f02a4cc9eae7f2b978416064ccd4c083638e3741abf9c34a7ad15f9f0  ./NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb" | shasum -a 256 --check
	apt install ./NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb -y
	;;
    *)
	echo "Invalid option. Expected [aarch|x86]."
	exit -1
	;;
esac
