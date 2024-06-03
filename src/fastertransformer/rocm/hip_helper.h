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

#include <hip/hip_runtime.h>
#include <hip/runtime_api.h>
#include <rocm_smi/rocm_smi.h>


#ifndef checkHipErrors
#define checkHipErrors(err) __checkHipErrors(err, __FILE__, __LINE__)

inline void __checkHipErrors(hipError_t err, const char *file, const int line) {
  if (HIP_SUCCESS != err) {
    const char *errorStr = hipGetErrorString(err);
    fprintf(stderr,
            "checkHipErrors() HIP API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}
#endif

#ifndef checkRsmiErrors
#define checkRsmiErrors(err) __checkRsmiErrors(err)

inline void __checkRsmiErrors(rsmi_status_t status){
  if (RSMI_STATUS_SUCCESS != status){
    const char* errorStr = rsmi_status_string(status); 
    fprintf(stderr, "Rsmi Runtime Error: %s \n", errorStr);
    exit(EXIT_FAILURE);
  }
}

#define sync_check_rocm_error() syncAndCheck(__FILE__, __LINE__)

inline void syncAndCheck(const char* const file, int const line){
  hipDeviceSynchronize();
  hipError_t result = hipGetLastError();
  if (result){
    throw std::runtime_error(std::string("[ERROR] ROCM runtime error: ") + (hipGetErrorString(result)) + " "
                             + file + ":" + std:to_string(line) + " \n")
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

