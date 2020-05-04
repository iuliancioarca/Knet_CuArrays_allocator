npics     = 512 # number of pics to use for traning
newdim    = 416 # new size for pic upscaling; if you modify this you need to modify the network
# Load MNIST data:
include(Knet.dir("data","mnist.jl"))
# load raw MNIST: 60_000 pics
_mnist_xtrn,_mnist_ytrn,_mnist_xtst,_mnist_ytst = mnist()
# Select only a few pics
_mnist_xtrn = _mnist_xtrn[:,:,:,1:npics]
_mnist_ytrn = _mnist_ytrn[1:npics]
_mnist_xtst = _mnist_xtst[:,:,:,1:npics]
_mnist_ytst = _mnist_ytst[1:npics]
# Hacky upscale using Images.jl
_mnist_xtrn = imresize(_mnist_xtrn[:,:,1,:],newdim,newdim)
_mnist_xtrn = reshape(_mnist_xtrn,newdim,newdim,1,npics)
_mnist_xtst = imresize(_mnist_xtst[:,:,1,:],newdim,newdim)
_mnist_xtst = reshape(_mnist_xtst,newdim,newdim,1,npics)
