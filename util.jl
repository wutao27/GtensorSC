# --------------- These codes implement some utility functions for general tensor spectral clustering ---------------------

# --------------- Author: Tao Wu
#---------------- Email: wutao27@gmail.com

# to modify
# readtensor skip

push!(LOAD_PATH, pwd())

using mymatrixfcn
import Base.isless

include("shift_fixpoint.jl")
include("tensor_cut.jl")

GAMA = 0.2; 

type algPara
  ALPHA::Float64
  MIN_NUM::Int64
  MAX_NUM::Int64
  PHI::Float64
end

# --------------- DataType to store the cuts -------------------------
if !isdefined(:cutTree)
  type cutTree
    n::Int64  #size of clustering
    cutValue::Float64  # the cutValue from sweep_cut
    Ptran::Float64  # the probability from the cluster to the other cluster
    invPtran::Float64 # the probabitlity from the other cluster to this cluster
    Pvol::Float64  #volumn of the nodes in the cluster (The probability in the cluster)
    subInd::Array{Int64,1} # nodes indices in this group
    tenInd::Array{Int64,1} # tensor indices(from parent) in this group
    data::Array{Any} # tensor data
    left::Union{cutTree, Void} # left child
    right::Union{cutTree, Void} # right child
  end
end
isless(a::cutTree,b::cutTree) = a.n > b.n

cutTree() = cutTree(0,0,0,0,0,Int64[],Int64[],Int64[],nothing,nothing)

function generate_treeNodes(parentNode::cutTree, permEv, cutPoint, cutValue, para)
  leftNode = cutTree(); rightNode = cutTree()
  leftNode.n = cutPoint; rightNode.n = parentNode.n - cutPoint
  leftNode.cutValue = cutValue; rightNode.cutValue = cutValue
  leftNode.Ptran = para[1]; rightNode.Ptran = para[2]
  leftNode.invPtran = para[2]; rightNode.invPtran = para[1]
  leftNode.Pvol = para[3]; rightNode.Pvol = 1-para[3]
  leftNode.left = nothing; rightNode.left = nothing
  leftNode.right = nothing; rightNode.right = nothing
  for i=1:cutPoint
    push!(leftNode.subInd, parentNode.subInd[ permEv[i] ])
    push!(leftNode.tenInd, permEv[i])
  end
  for i=cutPoint+1:length(permEv)
    push!(rightNode.subInd, parentNode.subInd[ permEv[i] ])
    push!(rightNode.tenInd, permEv[i])
  end

  leftNode.data = cut_tensor(parentNode, leftNode)
  rightNode.data = cut_tensor(parentNode, rightNode)

  if cutPoint>=parentNode.n - cutPoint || (para[1]==0 && para[2]==0 && para[3]==0)
    parentNode.left = leftNode; parentNode.right = rightNode
    parentNode.subInd = Int64[]; parentNode.tenInd = Int64[]; parentNode.data = []
    return (leftNode, rightNode)
  else
    parentNode.left = rightNode; parentNode.right = leftNode
    parentNode.subInd = Int64[]; parentNode.tenInd = Int64[]; parentNode.data = []
    return (rightNode, leftNode)
  end
end

# ---------------------------------------------------------------

# function to pass to mymatrixfcn
# make Adat row stocastic
function mymult(output,b,Adat,xT)
	e = ones(size(b,1))
  output = Adat*b + (xT*b)[1]*(e-Adat*e)
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
# format assumption: the first two rows are non-data
function read_tensor(abPath::AbstractString)
  f = readdlm(abPath, skipstart=0)
  m = size(f,2)
  P = Array[round(Int32,vec(f[:,1]))]
  for i = 2:m-1
      P = [P; Array[round(Int32,vec(f[:,i]))]]
  end
  P = [P; Array[map(Float64,vec(f[:,m]))]]
  f = 0
  return P
end

# normalize the tensor
function norm_tensor(P)
  n = length(P[1]); m = size(P,1)-1
  tab = Dict()
  for k=1:length(P[1])
    tempArray = []
    for i = 2:m
      push!(tempArray,P[i][k])
    end
    tempKey = tuple(tempArray...)
    if haskey(tab,tempKey)
      tab[tempKey] = tab[tempKey] + P[m+1][k]
    else
      tab[tempKey] = P[m+1][k]
    end
  end

  for k=1:length(P[1])
    tempArray = []
    for i = 2:m
      push!(tempArray,P[i][k])
    end
    tempKey = tuple(tempArray...)
    P[m+1][k] = P[m+1][k]/tab[tempKey]
  end
end

# compute the eigenvector by calling shift_fix, tran_matrix and eigs
function compute_egiv(P, al, ga)
  n = Int64(maximum(P[1]))
  v = ones(n)/n
  println("\tcomputing the super-spacey random surfer vector")
  x =shift_fix(P,v,al,ga,n)
  xT = transpose(x)
  println("\tgenerating transition matrix: P[x]")
  RT = tran_matrix(P, x)
  A = MyMatrixFcn{Float64}(n,n,(output, b) -> mymult(output, b, RT, xT))
  println("\tsolving the egenvector problem for P[x]")
  (ed, ev, nconv, niter, nmult, resid) = eigs(A,ritzvec=true,nev=2,which=:LM)
  return (ev,RT,x)
end


# function for generate a subtensor from parentNode
function cut_tensor(parentNode, treeNode::cutTree)
  n = Int64(parentNode.n)
  P = parentNode.data
  nz = size(P[1],1) # number of non-zeros
  m = size(P,1)-1  # tensor dimension
  tempInd = treeNode.tenInd

  tempDic = [tempInd[i] => i for i = 1:length(tempInd)] # new indices map
  validInd = falses(n)
  for item in tempInd
    validInd[item]=true
  end

  flag = trues(nz)  # whether to keep the entry in newT
  for i = 1:nz
    for j = 1:m
      if !validInd[ P[j][i] ]
        flag[i] = false; continue
      end
    end
  end

  newP = Array[P[1][flag]]
  for i=2:m+1
    push!(newP,P[i][flag])
  end

  for i=1:length(newP[1]) # changing to new column indices
    for j = 1:m
      newP[j][i] = tempDic[newP[j][i]]
    end
  end
  return newP
end

# check if there is empty indices in tensor P
function refine(P, treeNode::cutTree)
  allIndex = zeros(treeNode.n)
  for i=1:length(P[1])
    allIndex[P[1][i]] = 1
  end
  permIndex = sortperm(allIndex)
  if allIndex[permIndex[1]]==1 || length(P[1])==0
    return P
  end

  println("\tprocess empty indices in sub-tensor")
  cutPoint = 1
  while allIndex[permIndex[cutPoint]]==0
    cutPoint = cutPoint + 1
  end

  (t1,t2) = generate_treeNodes(treeNode, permIndex, cutPoint-1, 0, [0,0,0])
  return t2.data
end

# generate recursive two-way cuts for the tensor
# P is tensor data
# algParameters contains parameters for the algorithm: ALPHA, MIN_NUM, MAX_NUM, PHI
function tensor_speclustering(P, algParameters::algPara)
  rootNode = cutTree();rootNode.n = Int64(maximum(P[1]))
  rootNode.subInd = [ii for ii=1:rootNode.n] ; rootNode.tenInd = [ii for ii=1:rootNode.n]
  norm_tensor(P); rootNode.data = P
  h = Collections.heapify([cutTree(),cutTree()])
  dumyP = P

  for i = 1:typemax(Int64)
    println("\n-----------calculating #$i cut----------")
    if i!=1
      hp = Collections.heappop!(h)
      if hp.n <= algParameters.MIN_NUM
        println("completed recursive two-way cut");
        Collections.heappush!(h,hp);
        return (rootNode, h)
      end
      #dumyP = cut_tensor(dumyP, hp)
      dumyP = refine(hp.data, hp)
      if length(dumyP[1]) ==0 || maximum(dumyP[1]) <= algParameters.MIN_NUM
        println("tensor size smaller than MIN_NUM")
        continue
      end
    end

    println("\ttensor size $(maximum(dumyP[1])) with $(length(dumyP[1])) non-zeros")
    (ev,RT,x) = compute_egiv(dumyP,algParameters.ALPHA,GAMA)
    permEv = sortperm(real(ev[:,2]))
    println("\tgenerating the sweep_cut")
    (cutPoint, cutArray, cutValue, para) = sweep_cut(RT, x, permEv)

    if cutValue > algParameters.PHI && maximum(dumyP[1])<algParameters.MAX_NUM
      println("did not cut the tensor as biased conductance ($cutValue) > PHI");
      continue
    end
    if i==1
      (t1,t2) = generate_treeNodes(rootNode, permEv, cutPoint, cutValue, para)
      println("-- split $(rootNode.n) into $(t1.n) and $(t2.n) --")
      #print_figure(cutArray,"$(rootNode.n)_$(t1.n)_$(t2.n).png")
    else
      if hp.n > length(cutArray)+1
        hp = hp.right
      end
      (t1,t2) = generate_treeNodes(hp, permEv, cutPoint, cutValue, para)
      println("-- split $(hp.n) into $(t1.n) and $(t2.n) --")
      assert(hp.n == length(cutArray)+1)
      #print_figure(cutArray,"$(hp.n)_$(t1.n)_$(t2.n).png")
    end
    Collections.heappush!(h,t1);Collections.heappush!(h,t2)
  end
  return (rootNode, h)
end

# print k words from treeNode
function print_words(wordDic, treeNode::cutTree, k, sem)
  tempInd = treeNode.subInd
  println("-------------------- semantic value is $(sem) -----------------")
  println("cut parameters are: Ptran = $(treeNode.Ptran), invPtran = $(treeNode.invPtran), Pvol = $(treeNode.Pvol)")
  println("number of total words are $(treeNode.n)")
  for i=1:minimum([k,length(tempInd)])
    println(wordDic[tempInd[i]])
  end
  println("--------------------")
end

# generate the association matrix for clusters
function asso_matrix(P, rootNode::cutTree)
  m = size(P,1)-1
  indVec = zeros(Int64,rootNode.n)
  println("traverse tree");
  traCount = trav_tree(rootNode, indVec, 1)
  println("generating association matrix")
  assert(traCount - 1 == maximum(indVec))
  mat = zeros(traCount-1, traCount-1)
  for i = 1:size(P[1],1)
    col1 = indVec[P[1][i]]; col2 = indVec[P[2][i]]
    if col1 == col2
      continue
    end
    val = P[m+1][i];
    mat[col1, col1] += val
    mat[col2, col1] += val
  end
  d = Diagonal(mat)
  mat = (mat - d)*pinv(d)
  return mat
end

# compute popularity score
function group_score(P, rootNode::cutTree)
  mat=asso_matrix(P,r);
  n = size(mat,1)
  A = MyMatrixFcn{Float64}(n,n,(output, b) -> mymult2(output, b, sparse(mat)))
  (ed, ev, nconv, niter, nmult, resid) = eigs(A,ritzvec=true,nev=1,which=:LM);
  gscore = real(ev[:,1])
  gscore = gscore/sum(gscore)
  return gscore
end

# traverse the result cutTree
function trav_tree(treeNode::cutTree, indVec, startInd::Integer)
  if treeNode.left == nothing && treeNode.right == nothing
    tempInd = treeNode.subInd

    for i in tempInd
      indVec[i] = startInd
    end
    return startInd+1

  else
    newStart = trav_tree(treeNode.left, indVec, startInd)
    return trav_tree(treeNode.right, indVec, newStart)
  end
end

# print clutering result
function print_tree_word(treeNode::cutTree, semVec, startInd::Integer, wordDic; semTol_low = 0, semTol_up = 1, numTol_low = 0, numTol_up = 100)
  if treeNode.left == nothing && treeNode.right == nothing
    if semVec[startInd]>semTol_low && semVec[startInd]<semTol_up && treeNode.n < numTol_up && treeNode.n > numTol_low
      print_words(wordDic, treeNode, 100, semVec[startInd])
    end
    return startInd+1

  else
    newStart = print_tree_word(treeNode.left, semVec, startInd, wordDic, semTol_low = semTol_low, semTol_up = semTol_up, numTol_low = numTol_low, numTol_up = numTol_up)
    return print_tree_word(treeNode.right, semVec, newStart, wordDic, semTol_low = semTol_low, semTol_up = semTol_up, numTol_low = numTol_low, numTol_up = numTol_up)
  end
end