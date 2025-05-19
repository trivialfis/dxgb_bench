"""Simple script for querying C2C information."""

import pynvml as nm  # mamba install nvidia-ml-py -c rapidsai -c nvidia

nm.nvmlInit()
drv = nm.nvmlSystemGetDriverVersion()
print("Driver:", drv)

hdl = nm.nvmlDeviceGetHandleByIndex(0)
try:
    info = nm.nvmlDeviceGetC2cModeInfoV1(hdl)
except nm.NVMLError_NotSupported:
    info = None

# NVML_FI_DEV_C2C_LINK_GET_MAX_BW: C2C Link Speed in MBps for active links.
if info is not None and info.isC2cEnabled == 1:
    lc, bw = nm.nvmlDeviceGetFieldValues(
        hdl, [nm.NVML_FI_DEV_C2C_LINK_COUNT, nm.NVML_FI_DEV_C2C_LINK_GET_MAX_BW]
    )
    print("Link count:", lc.value.siVal, "Bandwidth:", bw.value.sllVal)

nm.nvmlShutdown()
