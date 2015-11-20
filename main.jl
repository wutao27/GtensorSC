include("util.jl")

using Gadfly
using Cairo

# read the transition tensor for fix_point and egenvectors
P = read_tensor("/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/data/raw/tensor_w3_.txt")

# read the raw tensor for the sweep_cut
T =read_tensor("/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/data/raw/raw_tensor_w3_.txt")

FILE_RUNTIME = open(RUNTIME_LOG_PATH,"a");FILE_RESULT = open(RESULT_LOG_PATH,"a")
close(FILE_RESULT);close(FILE_RUNTIME)


(r,h) = tensor_speclustering(T, P, 1, 0.6)
(r,h) = tensor_speclustering(T,P,1, 0.6, heapNodes = h, rNode = r)

indmin(cutArray[2000:length(cutArray)])
cutArray[30655]
#Plot to show the sweep_cut curve
print_figure(cutArray,"cut16.png")
cutArray

# show words
f = readdlm("/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/data/dic_w3_.txt", skipstart=2)
wordDic = [f[i,2] => f[i,1] for i = 1:size(f,1)]
for i=30655:length(permEv)
  println(wordDic[permEv[i]])
end

print_words(wordDic,r,r.right.left.left.right.right.left.right.right.left.right.right.right.right.right.right.left,200)

