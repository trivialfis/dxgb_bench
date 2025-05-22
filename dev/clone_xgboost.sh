INSTALL_XGBOOST=$1

if [[ -n ${INSTALL_XGBOOST} ]]; then
    cd /ws
    git clone --recursive  https://github.com/dmlc/xgboost.git
    cd xgboost
    git remote add jiamingy https://github.com/trivialfis/xgboost.git
    git fetch jiamingy ext-device-page
    git checkout ext-device-page
    cd /
fi
