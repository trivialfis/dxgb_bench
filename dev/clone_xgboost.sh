INSTALL_XGBOOST=$1

if [[ -n ${INSTALL_XGBOOST} ]]; then
    cd /ws
    git clone --recursive  https://github.com/dmlc/xgboost.git
    cd xgboost
    git checkout 6234b615a51c67193f001ea698ebcce7edb9d764
    cd /
fi
