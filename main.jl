include("util.jl")

using Gadfly

# read the transition tensor for fix_point and egenvectors
P = read_tensor("/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/data/tensor_w3_.txt")

# read the raw tensor for the sweep_cut
T =read_tensor("/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/data/raw_tensor_w3_.txt")
FILE_RUNTIME = open(RUNTIME_LOG_PATH,"a")
FILE_RESULT = open(RESULT_LOG_PATH,"a")

(cutArray, r,h) = tensor_speclustering(T, P, 2)
(cutArray, r,h) = tensor_speclustering(T,P,2,heapNodes = h, rNode = r)

close(FILE_RUNTIME)
close(FILE_RESULT)
#Plot to show the sweep_cut curve
plot(x=1:int(n/100), y=cutArray[1:int(n/100)])
plot(x=int(n/100):int(n/20), y=cutArray[int(n/100):int(n/20)])

# show words
f = readdlm("/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/data/dic_w3_.txt", skipstart=2)
wordDic = [f[i,2] => f[i,1] for i = 1:size(f,1)]
print_words(wordDic, rootNode, rootNode.right.left.left, 20)

rootNode.right.left<rootNode.right.right

h = Collections.heapify([rootNode, rootNode.left, rootNode.right, rootNode.right.left, rootNode.right.right, rootNode.right.left.left, rootNode.right.left.right])

hp = Collections.heappop!(h)
