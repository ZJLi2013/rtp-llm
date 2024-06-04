#include "src/fastertransformer/device/amd_impl/RocmDevice.h"

int main()
{
    DeviceInitParams rocm_params;
    rocm_device = RocmDevice(rocm_params) ;
    rocm_device.init() ;
    rocm_device.getDeviceProperties();
    rocm_device.getDeviceStatus();
    rocm_device.syncAndCheck(); 

    return 0 ;
}