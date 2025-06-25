INSTALL_XGBOOST=$1

if [[ -n ${INSTALL_XGBOOST} ]]; then
    cd /ws
    git clone --recursive  https://github.com/trivialfis/xgboost.git
    cd xgboost
    git checkout ext-compr
    cd /
fi
