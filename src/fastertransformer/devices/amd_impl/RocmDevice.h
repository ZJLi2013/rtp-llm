#pragma once 

#include <stdint.h>

class RocmDevice : public DeviceBase {
    public:
        RocmDevice(const DeviceInitParams& params);
        ~RocmDevice(); 

    public:
        void init() override ; 
        DeviceProperties getDeviceProperties() override ; 
        DeviceStatus getDeviceStatus() override ;
        // allocator api
        // TODO:
        void syncAndCheck() override ; 
    
    private: 

        hipStream_t stream_ ; 
        hipblasHandle_t hipblas_handle_ ; 
        hipDeviceProp_t device_prop_ ;
        BufferPtr rocrandstate_buf_; // for sampler use.
}