# Knet_CuArrays_allocator
Speed and memory footprint of Knet vs CuArrays allocators
```
#cd(raw"C:\Framework\Julia\Knet_CuArrays_allocator")
#activate .
include("test.jl")
```
modify ```Knet.cuallocator() = false``` to use Knet allocator
