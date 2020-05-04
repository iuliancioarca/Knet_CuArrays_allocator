#cd(raw"C:\Framework\Julia\Knet_CuArrays_allocator")
#activate .
using Knet
using Images
using Serialization
## Switch between Knet allocator and CuArrays allocator
Knet.cuallocator() = false
npics     = 32 # number of pics to use for traning
newdim    = 416 # new size for pic upscaling; if you modify this you need to modify the network
batchsize = 2
xtype     = (Knet.gpu()>=0 ? Knet.KnetArray{Float32} : Array{Float32})
##
include("types.jl")
include("load_prepare.jl")
# generate dataset for training and testing
dtrn = Knet.minibatch(_mnist_xtrn, _mnist_ytrn, batchsize; xtype=xtype)
dtst = Knet.minibatch(_mnist_xtst, _mnist_ytst, batchsize; xtype=xtype)

## Define a huge network
model = Chain(
    Conv(3,3,1,16,1,1,relu), #416
    Pool(2,2,0),
    Conv(3,3,16,32,1,1,relu), #208
    Pool(2,2,0),
    Conv(3,3,32,64,1,1,relu), #104
    Pool(2,2,0),
    Conv(3,3,64,128,1,1,relu), #52
    Pool(2,2,0),
    Conv(3,3,128,256,1,1,relu), #26
    Pool(2,2,0),
    Conv(3,3,256,512,1,1,relu), #13
    Conv(3,3,512,1024,1,1,relu), #13
    Conv(3,3,1024,1024,1,1,relu),
    Conv(1,1,1024,256,1,1,relu),
    Conv(1,1,256,10,1,0,relu),
    Dense(2250,10,identity)
    )
# Configure optimizer
optimizer = sgd(model,repeat(dtrn,2);lr=0.01)
# Now train
progress!(optimizer)
# # accuracy doesn't matter
# accuracy(model, dtst)
# # save model to disk
# serialize("model.a",model)
