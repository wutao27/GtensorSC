# --------------- These codes implement some utility functions for tensor spectral clustering ---------------------
# --------------- The paper can be found at :http://arxiv.org/abs/1502.05058

# --------------- Author: Tao Wu
#---------------- Email: wutao27@gmail.com

push!(LOAD_PATH, "/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/")
RUNTIME_LOG_PATH = "/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/data/ignore_runtime.txt"
RESULT_LOG_PATH = "/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/data/ignore_log.txt"
FILE_RUNTIME = open(RUNTIME_LOG_PATH,"a")
FILE_RESULT = open(RESULT_LOG_PATH,"a")

using mymatrixfcn
import Base.isless

include("shift_fixpoint.jl")
include("tensor_cut.jl")

function message(f,str::String)
  write(f,str*"\n")
  println(str)
end

# --------------- DataType to store the cuts -------------------------
if !isdefined(:cutTree)
  type cutTree
    n::Int64  #size of clustering
    cutValue::Float64  # the cutValue from sweep_cut
    TN::Float64  #sum of the tensor before the cut
    Tin::Float64  #sum of the tensor within the cluster
    Tout::Float64  #sum of the tensor outside the cluster
    Tvol::Float64  #volumn of the nodes in the cluster
    ind::BitArray  #ind[i]=true indicates i-th vertex from the parent tree node is in this cluster
    left::Union(cutTree, Nothing)
    right::Union(cutTree, Nothing)
    path::BitArray  # path from the root (e.g., (true, ture, false) means root->left->left->right)
  end
end
isless(a::cutTree,b::cutTree) = a.n > b.n

cutTree() = cutTree(0,0,0,0,0,0,BitArray(1),nothing,nothing,BitArray(1))

function generate_treeNodes(parentNode::cutTree, permEv, cutPoint, cutValue, para)
  leftNode = cutTree(); rightNode = cutTree()
  leftNode.n = cutPoint; rightNode.n = parentNode.n - cutPoint
  leftNode.cutValue = cutValue; rightNode.cutValue = cutValue
  leftNode.TN = para[1]; rightNode.TN = para[1]
  leftNode.Tin = para[2]; rightNode.Tin = para[3]
  leftNode.Tout = para[3]; rightNode.Tout = para[2]
  leftNode.Tvol = para[4]; rightNode.Tvol = para[1]-para[4]
  leftNode.left = nothing; rightNode.left = nothing
  leftNode.right = nothing; rightNode.right = nothing
  leftNode.path = copy(parentNode.path); rightNode.path = copy(parentNode.path)
  push!(leftNode.path, true); push!(rightNode.path, false)
  leftNode.ind = falses(parentNode.n); rightNode.ind = trues(parentNode.n)
  for i=1:cutPoint
    leftNode.ind[ permEv[i] ] = true
    rightNode.ind[ permEv[i] ] = false
  end
  parentNode.left = leftNode; parentNode.right = rightNode
  return (leftNode, rightNode)
end

# ---------------------------------------------------------------

# function to pass to mymatrixfcn
function mymult(output,b,Adat)
	e = ones(size(b,1))
  output = Adat*b + sum(b)/(size(b,1))*(e-Adat*e)
  return output
end


# load the tensor data
function read_tensor(abPath::String)
  f = readdlm(abPath, skipstart=2)
  m = size(f,2)
  P = Array[int32(vec(f[:,1]))]
  for i = 2:m-1
      P = [P, Array[int32(vec(f[:,i]))]]
  end
  P = [P, Array[float64(vec(f[:,m]))]]
  f = 0
  return P
end


# compute the eigenvector by calling shift_fix, tran_matrix and eigs
function compute_egiv(P, al, ga)
  n = int64(maximum(P[1]))
  v = ones(n)/n
  message(FILE_RUNTIME, "\tsolving the fix-point problem")
  x =@time shift_fix(P,v,al,ga,n)
  # compute the second left egenvector
  message(FILE_RUNTIME, "\tgenerating transition matrix")
  RT = @time tran_matrix(P, x)
  A = MyMatrixFcn{Float64}(n,n,(output, b) -> mymult(output, b, RT))
  message(FILE_RUNTIME,"\tsolving the egenvector problem")
  (ed, ev, nconv, niter, nmult, resid) = @time eigs(A,ritzvec=true,nev=2,which=:LM)
  return ev
end


# function for generate a subtensor based on cutPoint
# T is the original tensor
# currentMap is the array thats maps current indices to the original indices
# permEv is the permutation of sorted egenvector
function cut_tensor(T, treeNode::cutTree, rootNode::cutTree)
  n = rootNode.n
  nz = size(T[1],1) # number of non-zeros
  m = size(T,1)-1  # tensor dimension
  tempInd = [i for i=1:n]
  head = rootNode
  for i=2:length(treeNode.path)
    if treeNode.path[i]
      head = head.left
    else
      head = head.right
    end
    tempInd = tempInd[head.ind]
  end

  tempDic = [tempInd[i] => i for i = 1:length(tempInd)] # new indices map
  validInd = falses(n)
  for item in tempInd
    validInd[item]=true
  end

  flag = trues(nz)  # whether to keep the entry in newT
  for i = 1:nz
    for j = 1:m
      if !validInd [ T[j][i] ]
        flag[i] = false; continue
      end
    end
  end

  newT = Array[T[1][flag]]
  for i=2:m+1
    push!(newT,T[i][flag])
  end

  for i=1:length(newT[1]) # changing to new column indices
    for j = 1:m
      newT[j][i] = tempDic[newT[j][i]]
    end
  end
  return newT
end


# generate numCuts for the tensor
# T is the raw_tensor, P is the normalized tensor (column stochastic)
# heapNodes and rNode are optional, if given the algorithm will continue from where it is left over
function tensor_speclustering(T, P, numCuts::Integer; heapNodes = nothing, rNode = cutTree())
  alpha = 0.95; gama = 0.2; cutArray = 0; permEv = 0
  if heapNodes!=nothing
    h = heapNodes; rootNode = rNode
  else
    nowTime = strftime(time())
    write(FILE_RESULT,"\n+++++++++++++++++++++++++++++++"*nowTime*"+++++++++++++++++++++++++\n")
    write(FILE_RUNTIME,"\n+++++++++++++++++++++++++++++++"*nowTime*"+++++++++++++++++++++++++\n")
    rootNode = cutTree();rootNode.n = int64(maximum(P[1]))
    h = Collections.heapify([cutTree(),cutTree()])
    dumyT = T;dumyP = P
  end

  for i = 1:numCuts
    message(FILE_RUNTIME,"-----------calculating #$i cut----------")
    if i!=1 || heapNodes!=nothing
      hp = Collections.heappop!(h)
      dumyP = @time cut_tensor(P, hp, rootNode)
      if length(dumyP[1])==0
        message(FILE_RUNTIME,"no more cut");write(FILE_RESULT, "\nno more cut")
        Collections.heappush!(h,hp);
        error("There is no more cut")
      end
      dumyT = @time cut_tensor(T, hp, rootNode)
    end
    ev = compute_egiv(dumyP,alpha,gama)
    permEv = sortperm(real(ev[:,2]))
    message(FILE_RUNTIME,"\t generating the sweep_cut")
    (cutPoint, cutArray, cutValue, para) = @time sweep_cut(dumyT, permEv)
    if i==1 && heapNodes == nothing
      (t1,t2) = generate_treeNodes(rootNode, permEv, cutPoint, cutValue, para)
      message(FILE_RESULT, "\nsplit $(rootNode.n) into $(t1.n) and $(t2.n)")
    else
      (t1,t2) = generate_treeNodes(hp, permEv, cutPoint, cutValue, para)
      message(FILE_RESULT, "\nsplit $(hp.n) into $(t1.n) and $(t2.n)")
    end
    Collections.heappush!(h,t1);Collections.heappush!(h,t2)
  end
  return (permEv, cutArray, rootNode, h)
end

# print k words from treeNode
function print_words(wordDic, rootNode::cutTree, treeNode::cutTree, k)
  n = rootNode.n
  tempInd = [i for i=1:n]
  head = rootNode
  for i=2:length(treeNode.path)
    if treeNode.path[i]
      head = head.left
    else
      head = head.right
    end
    tempInd = tempInd[head.ind]
  end
  message(FILE_RESULT,"--------------------")
  message(FILE_RESULT, "cut parameters are: TN = $(treeNode.TN), Tin = $(treeNode.Tin), Tout = $(treeNode.Tout), Tvol = $(treeNode.Tvol)")
  message(FILE_RESULT,"number of total words are $(treeNode.n)")
  for i=1:minimum([k,length(tempInd)])
    message(FILE_RESULT,wordDic[tempInd[i]])
  end
  message(FILE_RESULT,"--------------------")
end

function print_figure(cutArray,file::String)
  n = length(cutArray)
  minPos = indmin(cutArray)
  if minPos< n/2
    ind = [1:2:2*minPos, (2*minPos+1):30:n]
  else
    ind = [1:30:(2*minPos-n), (2*minPos - n +1):2:n]
  end
  pl = plot(x=ind, y=cutArray[ind],Guide.xlabel("# nodes in group S"),Guide.ylabel("conductance cut"),Guide.title("Cut with $(n) nodes with $(minPos) and $(n-minPos)"))
  draw(PNG("/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/figure/"*file, 400, 130), pl)
end
