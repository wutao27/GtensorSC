# --------------- These codes implement some utility functions for general tensor spectral clustering ---------------------

# --------------- Author: Tao Wu
#---------------- Email: wutao27@gmail.com

# to modify
# path; ind
# readtensor skip
# println
# cut_tensor
push!(LOAD_PATH, pwd())

using mymatrixfcn
import Base.isless

include("shift_fixpoint.jl")
include("tensor_cut.jl")

ALPHA = 0.8; GAMA = 0.2; 
MIN_NUM = 5;
MAX_NUM = 100;

# --------------- DataType to store the cuts -------------------------
if !isdefined(:cutTree)
  type cutTree
    n::Int64  #size of clustering
    cutValue::Float64  # the cutValue from sweep_cut
    Ptran::Float64  # the probability from the cluster to the other cluster
    invPtran::Float64 # the probabitlity from the other cluster to this cluster
    Pvol::Float64  #volumn of the nodes in the cluster (The probability in the cluster)
    subInd::Array{Int64,1}
    tenInd::Array{Int64,1}
    data::Array{Any}
    left::Union{cutTree, Void}
    right::Union{cutTree, Void}
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

  parentNode.left = leftNode; parentNode.right = rightNode
  parentNode.subInd = Int64[]; parentNode.tenInd = Int64[]; parentNode.data = []
  return (leftNode, rightNode)
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
  #println("in compute_egiv n is $(n)")
  v = ones(n)/n
  println("\tsolving the fix-point problem")
  x =shift_fix(P,v,al,ga,n)
  xT = transpose(x)
  # compute the second left egenvector
  println("\tgenerating transition matrix")
  RT = tran_matrix(P, x)
  A = MyMatrixFcn{Float64}(n,n,(output, b) -> mymult(output, b, RT, xT))
  println("\tsolving the egenvector problem")
  (ed, ev, nconv, niter, nmult, resid) = eigs(A,ritzvec=true,nev=2,which=:LM)
  return (ev,RT,x)
end


# function for generate a subtensor based on cutPoint
# T is the original tensor
# currentMap is the array thats maps current indices to the original indices
# permEv is the permutation of sorted egenvector
function cut_tensor(parentNode, treeNode::cutTree)
  n = Int64(parentNode.n)
  T = parentNode.data
  nz = size(T[1],1) # number of non-zeros
  m = size(T,1)-1  # tensor dimension
  tempInd = treeNode.tenInd

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

  return newT
 
end

function refine(T, treeNode::cutTree)
  allIndex = zeros(treeNode.n)
  for i=1:length(T[1])
    allIndex[T[1][i]] = 1
  end
  permIndex = sortperm(allIndex)
  if allIndex[permIndex[1]]==1 || length(T[1])==0
    return T
  end

  println("find empty indices in sub-tensor")
  cutPoint = 1
  while allIndex[permIndex[cutPoint]]==0
    cutPoint = cutPoint + 1
  end

  (t1,t2) = generate_treeNodes(treeNode, permIndex, cutPoint-1, 0, [0,0,0])
  return t2.data
end

# generate numCuts for the tensor
# P is the normalized tensor (column stochastic)
# heapNodes and rNode are optional, if given the algorithm will continue from where it is left over
function tensor_speclustering(P, thres::Float64)

  nowTime = Libc.strftime(time())
  rootNode = cutTree();rootNode.n = Int64(maximum(P[1]))
  rootNode.subInd = [ii for ii=1:rootNode.n] ; rootNode.tenInd = [ii for ii=1:rootNode.n]
  rootNode.data = P
  h = Collections.heapify([cutTree(),cutTree()])
  dumyP = P

  for i = 1:100000
    println("-----------calculating #$i cut----------")
    if i!=1
      hp = Collections.heappop!(h)
      if hp.n <= MIN_NUM
        println("no more cut");
        Collections.heappush!(h,hp);
        return (rootNode, h)
      end
      #dumyP = cut_tensor(dumyP, hp)
      dumyP = refine(hp.data, hp)
      if length(dumyP[1]) ==0 || maximum(dumyP[1]) <= MIN_NUM
        println("nearly empty tensor")
        continue
      end
    end

    (ev,RT,x) = compute_egiv(dumyP,ALPHA,GAMA)
    permEv = sortperm(real(ev[:,2]))
    println("dumP: $(maximum(dumyP[1])), len(permEv): $(length(permEv))")
    println("\t generating the sweep_cut")
    (cutPoint, cutArray, cutValue, para) = sweep_cut(RT, x, permEv)
    #if maximum(dumyP[1])<MAX_NUM && cutValue > MIN_PRO + maximum(dumyP[1])*(thres - MIN_PRO)/MAX_NUM
    if cutValue > thres && maximum(dumyP[1])<MAX_NUM
      println("cutValue ($cutValue) > thres");
      println("cutValue ($cutValue)> thres");
      continue
    end
    if i==1
      (t1,t2) = generate_treeNodes(rootNode, permEv, cutPoint, cutValue, para)
      println("\nsplit $(rootNode.n) into $(t1.n) and $(t2.n)")
      #print_figure(cutArray,"$(rootNode.n)_$(t1.n)_$(t2.n).png")
    else
      if hp.n > length(cutArray)+1
        hp = hp.right
      end
      (t1,t2) = generate_treeNodes(hp, permEv, cutPoint, cutValue, para)
      println("\nsplit $(hp.n) into $(t1.n) and $(t2.n)")
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

# function print_figure(cutArray,file::AbstractString)
#   n = length(cutArray)+1
#   minPos = indmin(cutArray)
#   stepSize1 = round(Int64,n/1200+1)
#   stepSize2 = round(Int64,n/10000+1)
#   if minPos< n/2
#     ind = [1:stepSize2:2*minPos; (2*minPos+1):stepSize1:(n-1)]
#   else
#     ind = [1:stepSize1:(2*minPos-n); (2*minPos - n +1):stepSize2:(n-1)]
#   end
#   pl = plot(x=ind, y=cutArray[ind],Guide.xlabel("# nodes in group S"),Guide.ylabel("conductance cut"),Guide.title("Cut with $(n) nodes with $(minPos) and $(n-minPos)"))
#   draw(PNG("/Users/hasayake/Dropbox/research/2015/08-27-ml-pagerank/figure/"*file, 400, 130), pl)
# end

function asso_matrix(T, rootNode::cutTree)
  m = size(T,1)-1  # tensor dimension
  indVec = zeros(Int64,rootNode.n)
  println("traverse tree");
  traCount = trav_tree(rootNode, indVec, 1)
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