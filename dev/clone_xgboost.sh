INSTALL_XGBOOST=$1

if [[ -n ${INSTALL_XGBOOST} ]]; then
    cd /ws
    git clone --recursive  https://github.com/dmlc/xgboost.git
    cd xgboost
    git checkout 94bb1da0422a4f6ba7e34dd5cbee951d26403b67
    cd /
fi
