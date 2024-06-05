#include "src/fastertransformer/devices/amd_impl/RocmDevice.h"

namespace fastertransformer { 

int main()
{
    fastertransformer::DeviceInitParams rocm_params;

    fastertransformer::RocmDevice rocm_device = fastertransformer::RocmDevice(rocm_params) ;
    rocm_device.init() ;
    rocm_device.getDeviceProperties();
    rocm_device.getDeviceStatus();
    rocm_device.syncAndCheck(); 

    return 0 ;
}

} 