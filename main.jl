include("shift_fixpoint.jl")
# a simple test case
n=3
v = ones(3)/3
P = Array[Int32[1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3],Int32[1,1,2,3,1,1,1,2,3,3,3,1,1,2,2,2,3,3],Int32[1,2,1,1,1,2,3,1,1,2,3,1,2,1,2,3,1,2],Float64[1/3,1/3,1/3,1/3,1/3,1/3,1,1/3,1/3,1/2,1,1/3,1/3,1/3,1,1,1/3,1/2]]
alpha = 0.9
gama = 0.8
x = shift_fix(P,v,alpha,gama,n)


# n-gram test case
f = readdlm("/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/data/tensor_w2_.txt", skipstart=2)
m = size(f,2)
P = Array[int32(vec(f[:,1]))]
for i = 2:m-1
    P = [P, Array[int32(vec(f[:,i]))]]
end
P = [P, Array[float64(vec(f[:,m]))]]
f = 0
alpha = 0.9
gama = 0.2
n = int64(maximum(P[1]))
v = ones(n)/n
x =@time shift_fix(P,v,alpha,gama,n)

maximum(x)

minimum(x)
