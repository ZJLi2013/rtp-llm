
#include "src/fastertransformer/devices/amd_impl/RocmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/rocm/allocator_rocm.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

static const size_t DEFAULT_MAX_BATCH_SIZE = 256;

RocmDevice::RocmDevice(const DeviceInitParams& params) : DeviceBase(params){
    //TODO: hip_log 
    checkHipErrors(hipSetDevice(device_id_)); 
    checkHipErrors(hipStreamCreate(&stream_)); 

    auto allocator_ptr = new Allocator<AllocatorType::ROCM>(device_id_);
    allocator_ptr->setStream(stream_);
    allocator_.reset(allocator_ptr);
    auto host_allocator_ptr = new Allocator<AllocatorType::ROCM_HOST>(device_id_);
    host_allocator_ptr->setStream(stream_);
    host_allocator_.reset(host_allocator_ptr);

    checkHipBlasErrors(hipblasCreate(&hipblas_handle_));
    // no  hipblasLT
    checkHipBlasErrors(hipblasSetStream(hipblas_handle_, stream_));
    checkHipErrors(hipGetDeviceProperties(&device_prop_, device_id_));
    // no gemm config 
    // TODO: nncl config
} 

RocmDevice::~RocmDevice(){
    rocrandstate_buf_.reset(); 
    checkHipErrors(hipStreamDestroy(stream_));
    checkHipBlasErrors(hipblasDestroy(hipblas_handle_));
    //TODO: nncl destroy
}

void RocmDevice::init(){
    DeviceBase::init();
    printf("max batch size: %d\n", init_params_.max_batch_size);
    rocrandstate_buf_ = allocateBuffer({init_params_.max_batch_size * sizeof(hiprandStateXORWOW_t)});
}

void RocmDevice::syncAndCheck(){
    hipDeviceSynchronize();
    sync_check_rocm_error(); 
}

DeviceProperties RocmDevice::getDeviceProperties(){
    static DeviceProperties* prop = nullptr ;
    if(prop == nullptr){
        prop = new DeviceProperties();
        prop->type = DeviceType::Rocm;
        prop->id = device_id_;
        // TODO: 
        prop->tp_rank = 0;
        prop->tp_size = 0;
    }
    return *prop; 
}

DeviceStatus RocmDevice::getDeviceStatus(){
    DeviceStatus status;

    size_t total_bytes;
    checkHipErrors(hipMemGetInfo(&status.device_memory_status.free_bytes, &total_bytes));
    status.device_memory_status.used_bytes = total_bytes - status.device_memory_status.free_bytes;

    const auto buffer_status = queryBufferStatus();
    status.device_memory_status.allocated_bytes = buffer_status.device_allocated_bytes;
    status.device_memory_status.preserved_bytes = buffer_status.device_preserved_bytes;
    status.host_memory_status.allocated_bytes = buffer_status.host_allocated_bytes;

    rocmUtilization_t utilization ; 
    auto ret = rocmDeviceGetUtilizationRates(device_id_, utilization);
    assert(ret==RSMI_STATUS_SUCCESS);
    status.device_utilization = (float)utilization.gpu; 

    return status; 
}

RTP_LLM_REGISTER_DEVICE(Rocm);


}; // namespace fastertransformer
