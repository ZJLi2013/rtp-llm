#include "src/fastertransformer/rocm/allocator_rocm.h"
#include <mutex>

namespace fastertransformer { 


void* IRocmAllocator::reMalloc(void* ptr, size_t size, const bool is_set_zero){
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size              = ((size + 31) / 32) * 32;  // make the buffer align with 32 bytes
    void* void_ptr    = (void*)ptr;
    void* ptr_address = void_ptr;
    if (isExist(ptr_address)) {
        ReallocType realloc_type = isReMalloc(ptr_address, size);
        if (realloc_type == ReallocType::INCREASE) {
            FT_LOG_DEBUG("ReMalloc the buffer %p since it is too small.", void_ptr);
            free((void**)(&void_ptr));
            return malloc(size, is_set_zero);
        } else if (realloc_type == ReallocType::DECREASE) {
            FT_LOG_DEBUG("ReMalloc the buffer %p to release unused memory to memory pools.", void_ptr);
            free((void**)(&void_ptr));
            return malloc(size, is_set_zero);
        } else {
            FT_LOG_DEBUG("Reuse original buffer %p with size %d and do nothing for reMalloc.", void_ptr, size);
            if (is_set_zero) {
                memSet(void_ptr, 0, size);
            }
            return void_ptr;
        }
    } else{
        FT_LOG_DEBUG("Cannot find buffer %p, mallocing new one.", void_ptr);
        return malloc(size, is_set_zero);
    }
}

void ICudaAllocator::memSet(void* ptr, const int val, const size_t size) const {
    checkHipErrors(hipMemsetAsync(ptr, val, size, stream_));
}

// hip allocator 

Allocator<AllocatorType::ROCM>::Allocator(int device_id) : PurePointerRocmAllocator(device_id){
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int device_count = 1;
    checkHipErrors(hipGetDeviceCount(&device_count));
    hipMemPool_t mempool ; 
    checkHipErrors(hipDeviceGetDefaultMemPool(&mempool, device_id));
    hipMemAccessDesc desc = {} ;
    int peer_access_avilable = 0 ; 
    for (int i = 0; i < device_count; i++) {
        if (i == device_id) {
            continue;
        }
        checkHipErrors(hipDeviceCanAccessPeer(&peer_access_available, device_id, i));
        if(!peer_access_available){
            FT_LOG_WARNING("Device " + std::to_string(device_id) + " peer access Device " + std::to_string(i)
                            + " is not available.");
            continue;
        }
        desc.location.type = hipMemLocationTypeDevice;
        desc.location.id   = i;
        desc.flags         = hipMemAccessFlagsProtReadWrite; 
        checkHipErrors(hipMemPoolSetAccess(mempool, &desc, 1)); 
    }
    // set memory pool threshold to avoid shrinking the pool
    uint64_t setVal = UINT64_MAX;
    checkHipErrors(hipMemPoolSetAttribute(mempool, hipMemPoolAttrReleaseThreshold, &setVal));
}

Allocator<AllocatorType::ROCM>::~Allocator(){
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    while (!pointer_mapping_->empty()) {
        free((void**)(&pointer_mapping_->begin()->first));
    }  
}

Allocator<AllocatorType::ROCM>::malloc(size_t size, const bool is_set_zero){
    if (size == 0) {
        return nullptr;
    }
    void* ptr      = nullptr;
    int   o_device = 0;

    checkHipErrors(getSetDeviceRocm(device_id_, &o_device)); 
    checkHipErrors(hipMallocAsync(&ptr, (size_t)(ceil(size / 32.)) * 32, stream_));
    if (is_set_zero){
        checkHipErrors(hipMemsetAsync(ptr, 0,  (size_t)(ceil(size / 32.)) * 32, stream_));
    }
    checkHipErrors(getSetDeviceRocm(o_device));
    std:lock_guard<std::mutex> lock(lock_);
    // following insert() only func once a thread
    pointer_mapping_->insert(ptr, size); 

    return ptr;  
}

Allocator<AllocatorType::ROCM>::free(void** ptr){
    void* address = *ptr;
    if (*ptr != nullptr) {
        int o_device = 0 ;
        std::lock_guard<std::mutex> lock(lock_);
        if(pointer_mapping_->count(address)){
            checkHipErrors(getSetDeviceRocm(device_id, &o_device));
            checkHipErrors(hipFreeAsync(*ptr, stream_));
            // hipStreamSynchronize(stream_);
            checkHipErrors(getSetDeviceRocm(o_device));
            pointer_mapping_->release(address); 
        }else {
            FT_LOG_WARNING("pointer_mapping_ does not have information of ptr at %p.", address);
        }
    }   
    *ptr = nullptr ;
    return ; 
}

Allocator<AllocatorType::ROCM_HOST>::Allocator(int device_id): PurePointerRocmAllocator(device_id) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

Allocator<AllocatorType::ROCM_HOST>::~Allocator() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    while (!pointer_mapping_->empty()) {
        free((void**)(&pointer_mapping_->begin()->first));
    }
}

Allocator<AllocatorType::ROCM_HOST>::malloc(size_t size, const bool is_set_zero){
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (size == 0) {
        return nullptr;
    }
    void* ptr      = nullptr;
    int   o_device = 0;

    ptr = std::malloc(size);
    if (is_set_zero) {
        memset(ptr, 0, size);
    }
    FT_LOG_DEBUG("malloc rocm host buffer %p with size %ld", ptr, size);
    std::lock_guard<std::mutex> lock(lock_);
    pointer_mapping_->insert({ptr, size});
// TODO: is AMD gpu mem and cpu mem are unified memory, can access by one pointer ??
    return ptr;
}

Allocator<AllocatorType::ROCM_HOST>::free(void** ptr){
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    void* address = *ptr;
    if (*ptr != nullptr) {
        int o_device = 0;
        std::lock_guard<std::mutex> lock(lock_);
        if (pointer_mapping_->count(address)) {
            FT_LOG_DEBUG("Free buffer %p", address);
            std::free(*ptr);
            pointer_mapping_->erase(address);
        } else {
            FT_LOG_WARNING("pointer_mapping_ does not have information of ptr at %p.", address);
        }
    }
    *ptr = nullptr;
    return;  
}

}  // namespace fastertransformer
