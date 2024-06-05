/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef SRC_FASTERTRANSFORMER_ROCM_HIP_HELPER_H_
#define SRC_FASTERTRANSFORMER_ROCM_HIP_HELPER_H_

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "rocm_smi/rocm_smi.h"
#include <hipblas/hipblas.h>


namespace fastertransformer {

#ifndef checkHipErrors
#define checkHipErrors(err) __checkHipErrors(err, __FILE__, __LINE__)

inline void __checkHipErrors(hipError_t err, const char *file, const int line) {
  if (hipSuccess != err) {
    const char *errorStr = hipGetErrorString(err);
    fprintf(stderr,
            "checkHipErrors() HIP API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}
#endif

// TODO
#ifndef checkHipBlasErrors
#define checkHipBlasErrors(err) __checkHipBlasErrors(err, __FILE__, __LINE__)

inline void __checkHipBlasErrors(hipblasStatus_t err, const char* file, const int line){
  if(HIPBLAS_STATUS_SUCCESS != err){
    fprintf(stderr, "checkHipBlasErrors() HIPBlas API error = %04d from file <%s>,"
             "line %i. \n",
              err, file, line);
    exit(EXIT_FAILURE); 
  }
}
#endif 

inline hipError_t getSetDeviceRocm(int i_device, int* o_device=nullptr){
    int         current_dev_id = 0;
    hipError_t err = hipErrorTbd ; 
    if (o_device != nullptr){
      err = hipGetDevice(&current_dev_id);
      if(err != hipSuccess){
        return err ;
      }
      if (current_dev_id == i_device){
        *o_device = i_device ;
      }else{
        err = hipSetDevice(i_device);
        if(err != hipSuccess){
          return err; 
        }
        *o_device = current_dev_id ;
      }
    }else{
      err = hipSetDevice(i_device);
      if(err != hipSuccess){
        return err; 
      }
    }
    return hipSuccess; 
}

#ifndef checkRsmiErrors
#define checkRsmiErrors(err) __checkRsmiErrors(err)

inline void __checkRsmiErrors(rsmi_status_t status){
  if (RSMI_STATUS_SUCCESS != status){
    // const char* errorStr = rsmi_status_string(status); 
    const char* errorStr = nullptr ; 
    rsmi_status_string(status, &errorStr);
    fprintf(stderr, "Rsmi Runtime Error: %s \n", errorStr);
    exit(EXIT_FAILURE);
  }
}
#endif 

#define sync_check_rocm_error() syncAndCheckRocm(__FILE__, __LINE__)

inline void syncAndCheckRocm(const char* const file, int const line){
  hipDeviceSynchronize();
  hipError_t result = hipGetLastError();
  if (result){
    throw std::runtime_error(std::string("[ERROR] ROCM runtime error: ") + (hipGetErrorString(result)) + " "
                             + file + ":" + std::to_string(line) + " \n");
  }
}

typedef struct {
  uint32_t memory ;
  uint32_t gpu; 
} rocmUtilization_t ; 

inline rsmi_status_t  rocmDeviceGetUtilizationRates(uint32_t device_id, rocmUtilization_t& utilization){
  // init rsmi 
  checkRsmiErrors(rsmi_init(0));
  // get GPU utilization
  checkRsmiErrors(rsmi_dev_busy_percent_get(device_id, &utilization.gpu));
  // get mem utilization
  checkRsmiErrors(rsmi_dev_memory_busy_percent_get(device_id, &utilization.memory));

  return RSMI_STATUS_SUCCESS ; 
}

}  // namespace fastertransformer

#endif 