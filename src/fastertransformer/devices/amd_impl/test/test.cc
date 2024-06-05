#include "src/fastertransformer/devices/amd_impl/RocmDevice.h"

using namespace fastertransformer ;

int main()
{
    DeviceInitParams rocm_params;
    RocmDevice rocm_device = RocmDevice(rocm_params) ;
    rocm_device.init() ;
    DeviceProperties props = rocm_device.getDeviceProperties();
    DeviceStatus dstatus = rocm_device.getDeviceStatus();
    rocm_device.syncAndCheck(); 

    return 0 ;
}

