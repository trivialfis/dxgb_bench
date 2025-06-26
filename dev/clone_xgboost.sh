INSTALL_XGBOOST=$1
XGBOOST_CHECKOUT=$2

echo "INSTALL_XGBOOST: ${INSTALL_XGBOOST}, XGBOOST_CHECKOUT: ${XGBOOST_CHECKOUT}"

if [[ -n ${INSTALL_XGBOOST} ]]; then
    cd /ws
    git clone --recursive  https://github.com/dmlc/xgboost.git
    cd xgboost
    if [[ -n ${XGBOOST_CHECKOUT} ]]; then
	echo "Checkout"
	git checkout ${XGBOOST_CHECKOUT}
    fi
    cd /
fi
