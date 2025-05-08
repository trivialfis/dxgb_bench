// g++ -isystem /usr/local/cuda/include ./test-cuda-alloc.cc  -L/usr/local/cuda/lib64/ -lcuda -lcudart -o test-cuda-alloc && ./test-cuda-alloc
/**
cudaHostAlloc:	0
cudaMallocHost:	0
cudaHostRegister:	0
cudaMalloc:	1
cudaMallocManaged:	0
mem_pool_ptr:	1
minimum granularity:	2097152
mem_create_ptr:	1
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string.h>

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      const char* err_str = cudaGetErrorString(rt); 	                       \
      std::cout << "Runtime API error \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::cout << err_str << std::endl;                                      \
      throw;                                                                   \
    }                                                                          \
  } while (0);

#define CU_CHECK(c) do { \
  CUresult err = (c); \
  if (err != CUDA_SUCCESS) { \
    const char* err_str;     \
    cuGetErrorString(err, &err_str); \
    std::cout << (#c) << std::endl; \
    throw std::runtime_error(std::string("CUDA Driver API error: '") + std::to_string(err) + "': " + err_str); \
  } \
} while(0)

template<typename T>
bool test_ptr(T* ptr) {
  bool test;
  try {
    CU_CHECK(cuPointerGetAttribute(&test, CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE, (CUdeviceptr)ptr));
  } catch(std::exception& e) {
    std::cerr << "error:" << e.what() << std::endl;
    test = false;
  }
  return test;
}

void test_cuda_malloc() {
  {
    // Initialize context
    CU_CHECK(cuInit(0));
  }
  // {
  //   // Pageable host memory
  //   char a[1024];
  //   std::cout << test_ptr(a) << std::endl;
  // }
  {
    char* ptr = static_cast<char*>(std::malloc(1024));
    cudaPointerAttributes attrs;
    CUDA_CHECK(cudaPointerGetAttributes(&attrs, ptr));
    std::cout << "malloc:\t" << attrs.type << std::endl;
  }
  {

    // pinned memory (through cudaHostAlloc)
    char* pinned_ptr;
    CUDA_CHECK(cudaHostAlloc(&pinned_ptr, 1024, cudaHostAllocDefault));
    std::cout << "cudaHostAlloc:\t" << test_ptr(pinned_ptr) << std::endl;
  }
  {
    // pinned memory (through cudaMallocHost)
    char* pinned_ptr_2;
    CUDA_CHECK(cudaMallocHost(&pinned_ptr_2, 1024));
    std::cout << "cudaMallocHost:\t" << test_ptr(pinned_ptr_2) << std::endl;
  }
  {
    char* pinned_ptr_2 = static_cast<char*>(std::malloc(1024));
    CUDA_CHECK(cudaHostRegister(pinned_ptr_2, 1024, cudaHostRegisterDefault));
    std::cout << "cudaHostRegister:\t" << test_ptr(pinned_ptr_2) << std::endl;
  }
  {
    // device memory
    char* device_ptr;
    CUDA_CHECK(cudaMalloc(&device_ptr, 1024));
    std::cout << "cudaMalloc:\t" << test_ptr(device_ptr) << std::endl;
  }
  {
    // unified memory
    char* unified_ptr;
    CUDA_CHECK(cudaMallocManaged(&unified_ptr, 1024));
    std::cout << "cudaMallocManaged:\t" << test_ptr(unified_ptr) << std::endl;
  }
  {
    cudaMemPoolProps props = {};
    props.location.type = cudaMemLocationTypeHostNuma;
    props.location.id = 0;
    props.allocType     = cudaMemAllocationTypePinned;
    props.usage         = cudaMemPoolCreateUsageHwDecompress;
    cudaMemPool_t mem_pool;
    CUDA_CHECK(cudaMemPoolCreate(&mem_pool, &props));
    char* mem_pool_ptr;
    CUDA_CHECK(cudaMallocFromPoolAsync(&mem_pool_ptr, 1024, mem_pool, 0));
    std::cout << "mem_pool_ptr:\t" << test_ptr(mem_pool_ptr) << std::endl;
  }
  {
    CUdeviceptr mem_create_ptr;
    CUmemGenericAllocationHandle allocHandle;
    CUmemAllocationProp props_2 = {};
    props_2.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    props_2.location.id = 0;
    props_2.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    props_2.allocFlags.usage = CU_MEM_CREATE_USAGE_HW_DECOMPRESS;
    size_t granularity;
    CU_CHECK(cuMemGetAllocationGranularity(&granularity, &props_2, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    std::cout << "minimum granularity:\t" << granularity << std::endl;

    // Create the allocation handle
    CU_CHECK(cuMemCreate(&allocHandle, granularity, &props_2, 0));

    // Reserve virtual address space
    CU_CHECK(cuMemAddressReserve(&mem_create_ptr, granularity, 0, 0, 0));

    // Map the physical memory to the virtual address
    CU_CHECK(cuMemMap(mem_create_ptr, granularity, 0, allocHandle, 0));

    // Set access permissions
    std::cout << "mem_create_ptr:\t" << test_ptr((char*)mem_create_ptr) << std::endl;
  }
}


int main() {
  test_cuda_malloc();
  return 0;
}
