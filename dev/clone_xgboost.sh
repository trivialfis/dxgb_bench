INSTALL_XGBOOST=$1
XGBOOST_REPO=$2
XGBOOST_CHECKOUT=$3

echo "INSTALL_XGBOOST: ${INSTALL_XGBOOST}, XGBOOST_REPO: ${XGBOOST_REPO} XGBOOST_CHECKOUT: ${XGBOOST_CHECKOUT}"

if [[ -n ${INSTALL_XGBOOST} ]]; then
    cd /ws
    if [[ -n ${XGBOOST_REPO} ]]; then
	git clone --recursive ${XGBOOST_REPO}
    else
	git clone --recursive  https://github.com/dmlc/xgboost.git
    fi

    cd xgboost

    if [[ -n ${XGBOOST_CHECKOUT} ]]; then
	echo "Checkout"
	git checkout ${XGBOOST_CHECKOUT}
    fi

    cd /
fi
