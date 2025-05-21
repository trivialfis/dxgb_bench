#!/usr/bin/env bash

set -euox pipefail

cd /ws
git clone https://github.com/gsauthof/cgmemtime.git
cd cgmemtime
make
cp cgmemtime /usr/bin/cgmemtime
cd /
