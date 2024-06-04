#pragma once 
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/rocm/hip_helper.h"
#include <hip/hip_runtime.h>
#include <unistd.h>

namespace fastertransformer {

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
        std::unique_ptr<IAllocator> allocator_;
        std::unique_ptr<IAllocator> host_allocator_;
        
        hipStream_t stream_ ; 
        hipblasHandle_t hipblas_handle_ ; 
        hipDeviceProp_t device_prop_ ;

        // BufferPtr rocrandstate_buf_; // for sampler use.
} ;

} // namespace fastertransformer
