#!/usr/bin/env bash

set -euox pipefail

INSTALL_XGBOOST=$1
XGBOOST_REPO=$2
XGBOOST_CHECKOUT=$3

echo "INSTALL_XGBOOST: ${INSTALL_XGBOOST}, XGBOOST_REPO: ${XGBOOST_REPO} XGBOOST_CHECKOUT: ${XGBOOST_CHECKOUT}"

if [[ -n ${INSTALL_XGBOOST} ]]; then
    cd /ws
    git clone "${XGBOOST_REPO:-https://github.com/dmlc/xgboost.git}" xgboost

    cd xgboost

    if [[ -n ${XGBOOST_CHECKOUT} ]]; then
	echo "Checkout"
	git checkout "${XGBOOST_CHECKOUT}"
    fi

    git submodule update --init --recursive

    cd /
fi
