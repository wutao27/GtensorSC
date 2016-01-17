# --------------- These codes implement some utility functions for tensor spectral clustering ---------------------
# --------------- The paper can be found at :http://arxiv.org/abs/1502.05058

# --------------- Author: Tao Wu
#---------------- Email: wutao27@gmail.com
# to modify
# cutTree definition
# generate_treeNodes
# tensor_spectral_cluster
# print_word
# sweep_cut

push!(LOAD_PATH, "/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/")
RUNTIME_LOG_PATH = "/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/data/log/ignore_runtime.txt"
RESULT_LOG_PATH = "/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/data/log/ignore_log.txt"
FILE_RUNTIME = open(RUNTIME_LOG_PATH,"a")
FILE_RESULT = open(RESULT_LOG_PATH,"a")

using mymatrixfcn
import Base.isless

include("shift_fixpoint.jl")
include("tensor_cut.jl")

function message(f,str::AbstractString)
  write(f,str*"\n")
  println(str)
end

# --------------- DataType to store the cuts -------------------------
if !isdefined(:cutTree)
  type cutTree
    n::Int64  #size of clustering
    cutValue::Float64  # the cutValue from sweep_cut
    #TN::Float64  #sum of the tensor before the cut
    #Tin::Float64  #sum of the tensor within the cluster
    Ptran::Float64  # the probability from the cluster to the other cluster
    invPtran::Float64 # the probabitlity from the other cluster to this cluster
    Pvol::Float64  #volumn of the nodes in the cluster (The probability in the cluster)
    ind::BitArray  #ind[i]=true indicates i-th vertex from the parent tree node is in this cluster
    left::Union{cutTree, Void}
    right::Union{cutTree, Void}
    path::BitArray  # path from the root (e.g., (true, ture, false) means root->left->left->right)
  end
end
isless(a::cutTree,b::cutTree) = a.n > b.n

cutTree() = cutTree(0,0,0,0,0,BitArray(1),nothing,nothing,BitArray(1))

function generate_treeNodes(parentNode::cutTree, permEv, cutPoint, cutValue, para)
  leftNode = cutTree(); rightNode = cutTree()
  leftNode.n = cutPoint; rightNode.n = parentNode.n - cutPoint
  leftNode.cutValue = cutValue; rightNode.cutValue = cutValue
  leftNode.Ptran = para[1]; rightNode.Ptran = para[2]
  leftNode.invPtran = para[2]; rightNode.invPtran = para[1]
  leftNode.Pvol = para[3]; rightNode.Pvol = 1-para[3]
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
# make Adat row stocastic
function mymult(output,b,Adat,xT)
	e = ones(size(b,1))
  output = Adat*b + (xT*b)*(e-Adat*e)
  return output
end

# function to pass to mymatrixfcn
# make Adat column stocastic
function mymult2(output,b,Adat)
  n = size(b,1)
  e = ones(n)
  tempOut = Adat*b
  output = tempOut + (sum(b) - sum(tempOut))/n*e
  return output
end

# load the tensor data
function read_tensor(abPath::AbstractString)
  f = readdlm(abPath, skipstart=2)
  m = size(f,2)
  P = Array[round(Int32,vec(f[:,1]))]
  for i = 2:m-1
      P = [P; Array[round(Int32,vec(f[:,i]))]]
  end
  P = [P; Array[map(Float64,vec(f[:,m]))]]
  f = 0
  return P
end


# compute the eigenvector by calling shift_fix, tran_matrix and eigs
function compute_egiv(P, al, ga)
  n = Int64(maximum(P[1]))
  println("in compute_egiv n is $(n)")
  v = ones(n)/n
  message(FILE_RUNTIME, "\tsolving the fix-point problem")
  x =shift_fix(P,v,al,ga,n)
  xT = transpose(x)
  # compute the second left egenvector
  message(FILE_RUNTIME, "\tgenerating transition matrix")
  RT = tran_matrix(P, x)
  A = MyMatrixFcn{Float64}(n,n,(output, b) -> mymult(output, b, RT, xT))
  message(FILE_RUNTIME,"\tsolving the egenvector problem")
  (ed, ev, nconv, niter, nmult, resid) = eigs(A,ritzvec=true,nev=2,which=:LM)
  return (ev,RT,x)
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
      if !validInd[ T[j][i] ]
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
  ttt = maximum(newT[1])
  bbb = length(tempInd)
  println("treeNode.n is $(treeNode.n) and len(newT) is $(ttt) and len(tempInd) is $(bbb)")
  return newT
end


# generate numCuts for the tensor
# P is the normalized tensor (column stochastic)
# heapNodes and rNode are optional, if given the algorithm will continue from where it is left over
function tensor_speclustering(P, numCuts::Integer, thres::Float64; heapNodes = nothing, rNode = cutTree())
  alpha = 0.95; gama = 0.2; 
  if heapNodes!=nothing
    #println("yes")
    h = heapNodes; rootNode = rNode
  else
    nowTime = Libc.strftime(time())
    write(FILE_RESULT,"\n+++++++++++++++++++++++++++++++"*nowTime*"+++++++++++++++++++++++++\n")
    write(FILE_RUNTIME,"\n+++++++++++++++++++++++++++++++"*nowTime*"+++++++++++++++++++++++++\n")
    rootNode = cutTree();rootNode.n = Int64(maximum(P[1]))
    h = Collections.heapify([cutTree(),cutTree()])
    dumyP = P
  end

  for i = 1:numCuts
    message(FILE_RUNTIME,"-----------calculating #$i cut----------")
    if i!=1 || heapNodes!=nothing
      hp = Collections.heappop!(h)
      if hp.n == 0
        message(FILE_RUNTIME,"no more cut");write(FILE_RESULT, "\nno more cut")
        Collections.heappush!(h,hp);
        return (rootNode, h)
      end
      dumyP = cut_tensor(P, hp, rootNode)
      if length(dumyP[1])==0
        message(FILE_RUNTIME,"empty tensor")
        continue
      end
    end
    (ev,RT,x) = compute_egiv(dumyP,alpha,gama)
    permEv = sortperm(real(ev[:,2]))
    println("dumP: $(maximum(dumyP[1])), len(permEv): $(length(permEv))")
    message(FILE_RUNTIME,"\t generating the sweep_cut")
    (cutPoint, cutArray, cutValue, para) = sweep_cut(RT, x, permEv)
    if cutValue > thres
      message(FILE_RUNTIME, "cutValue ($cutValue) > thres");
      message(FILE_RESULT, "cutValue ($cutValue)> thres");
      continue
    end
    if i==1 && heapNodes == nothing
      (t1,t2) = generate_treeNodes(rootNode, permEv, cutPoint, cutValue, para)
      message(FILE_RESULT, "\nsplit $(rootNode.n) into $(t1.n) and $(t2.n)")
      print_figure(cutArray,"$(rootNode.n)_$(t1.n)_$(t2.n).png")
    else
      (t1,t2) = generate_treeNodes(hp, permEv, cutPoint, cutValue, para)
      message(FILE_RESULT, "\nsplit $(hp.n) into $(t1.n) and $(t2.n)")
      assert(hp.n == length(cutArray)+1)
      print_figure(cutArray,"$(hp.n)_$(t1.n)_$(t2.n).png")
    end
    Collections.heappush!(h,t1);Collections.heappush!(h,t2)
  end
  return (rootNode, h)
end

# print k words from treeNode
function print_words(wordDic, rootNode::cutTree, treeNode::cutTree, k, sem)
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
  message(FILE_RESULT,"-------------------- semantic value is $(sem) -----------------")
  message(FILE_RESULT, "cut parameters are: Ptran = $(treeNode.Ptran), invPtran = $(treeNode.invPtran), Pvol = $(treeNode.Pvol)")
  message(FILE_RESULT,"number of total words are $(treeNode.n)")
  for i=1:minimum([k,length(tempInd)])
    message(FILE_RESULT,wordDic[tempInd[i]])
  end
  message(FILE_RESULT,"--------------------")
end

function print_figure(cutArray,file::AbstractString)
  n = length(cutArray)+1
  minPos = indmin(cutArray)
  stepSize1 = round(Int64,n/1200+1)
  stepSize2 = round(Int64,n/10000+1)
  if minPos< n/2
    ind = [1:stepSize2:2*minPos; (2*minPos+1):stepSize1:(n-1)]
  else
    ind = [1:stepSize1:(2*minPos-n); (2*minPos - n +1):stepSize2:(n-1)]
  end
  pl = plot(x=ind, y=cutArray[ind],Guide.xlabel("# nodes in group S"),Guide.ylabel("conductance cut"),Guide.title("Cut with $(n) nodes with $(minPos) and $(n-minPos)"))
  draw(PNG("/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/figure/"*file, 400, 130), pl)
end

function asso_matrix(T, rootNode::cutTree)
  m = size(T,1)-1  # tensor dimension
  indVec = zeros(Int32,rootNode.n)
  println("traverse tree");
  traCount = trav_tree(rootNode, rootNode, indVec, 1)
  println("generating association matrix")
  assert(traCount - 1 == maximum(indVec))
  mat = zeros(traCount-1, traCount-1)
  for i = 1:size(T[1],1)
    col1 = indVec[T[1][i]]; col2 = indVec[T[2][i]]
    if col1 == col2
      continue
    end
    val = T[m+1][i];
    mat[col1, col1] += val
    mat[col2, col1] += val
  end
  d = Diagonal(mat)
  mat = (mat - d)*pinv(d)
  return mat
end

function trav_tree(rootNode::cutTree, treeNode::cutTree, indVec, startInd::Integer)
  #println(treeNode.n)
  if treeNode.left == nothing && treeNode.right == nothing
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

    for i in tempInd
      indVec[i] = startInd
    end
    return startInd+1

  else
    newStart = trav_tree(rootNode, treeNode.left, indVec, startInd)
    return trav_tree(rootNode, treeNode.right, indVec, newStart)
  end
end

function print_tree_word(rootNode::cutTree, treeNode::cutTree, semVec, startInd::Integer, wordDic; semTol = 99999, numTol = 0)
  if treeNode.left == nothing && treeNode.right == nothing
    if semVec[startInd]<semTol && treeNode.n > numTol
      print_words(wordDic, rootNode, treeNode, 100, semVec[startInd])
    end
    return startInd+1

  else
    newStart = print_tree_word(rootNode, treeNode.left, semVec, startInd, wordDic, semTol = semTol, numTol = numTol)
    return print_tree_word(rootNode, treeNode.right, semVec, newStart, wordDic, semTol = semTol, numTol = numTol)
  end
end