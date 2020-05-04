# Define convolutional layer:
struct Conv; w; stride; padding; f;end #Define convolutional layer
(c::Conv)(x)= c.f.(conv4(c.w,x; stride = c.stride, padding = c.padding))
Conv(w1::Int,w2::Int,cx::Int,cy::Int,st,pd,f) = Conv(param(w1,w2,cx,cy),st,pd,f)

# Define dense layer:
struct Dense; w; b; f; end
(d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b)
Dense(i::Int,o::Int,f=relu) = Dense(param(o,i), param0(o), f)

# Define pooling layer:
struct Pool; size; stride; pad; end # Define pool layer
(p::Pool)(x) = pool(x; window = p.size, stride = p.stride, padding=p.pad) #pool function

# Define a chain of layers and a loss function:
struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x) #chain function
(c::Chain)(x,y) = nll(c(x),y) # loss function
