# --------------- These codes compute the multilinear pagerank problem ---------------------
#compute the kron of x
function compute_kron(P, index, x)
  m = size(P,1)-1
  result = 1
  for i in 1:m-1
    result = result*x[P[i+1][index]]
  end
  return result
end

# row_vec, col_vec, val_vec is a sparse n^m by n matrix, x is a n-dim vector
function sparse_kron(P, x)
  n = size(x,1); m = size(P,1)-1; Px = zeros(n)
  col = P[1][1];sumval = 0
  #println("debug")
  for ind = 1:size(P[1],1)
    if P[1][ind]!= col
      Px[col]=sumval
      sumval = P[m+1][ind]*compute_kron(P,ind,x)
      col = P[1][ind]
    else
      sumval += P[m+1][ind]*compute_kron(P,ind,x)
    end
    if ind==size(P[1],1)
      Px[col]=sumval
    end
  end
  return Px
end


# P is a sparse tensor with row(columns) list and value list
function shift_fix(P, v, alpha, gama, n)
  maxiter = 10000
  tol = 1/10^(8)
  x_old = rand(n)
  x_old = x_old/sum(x_old)
#  row_vec, col_vec, val_vec = findnz(P)
  for i = 1:maxiter
    Px = sparse_kron(P, x_old)
    x_new = (alpha/(1+gama))*Px + ((1-alpha)/(1+gama))*v + (gama/(1+gama))*x_old
    res = sum(abs(x_new-x_old))
    if res <= tol
      println("find the fix point within ",i," iteration")
      println("x = ", x_new)
      break
    end
    println("---iter ",i," residual is ",res)
    x_old = x_new
  end
end

# test case
n=3
P = sparse([1/3 1/3 1/3; 1/3 1/3 1/3; 1/3 1/3 1/3; 1/3 1/3 1/3; 0 0 1; 0 1/2 1/2; 0 1 0;0 0 1; 0 1 0])
v = ones(3)/3
P = Array[Int32[1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3],Int32[1,1,2,3,1,1,1,2,3,3,3,1,1,2,2,2,3,3],Int32[1,2,1,1,1,2,3,1,1,2,3,1,2,1,2,3,1,2],Float64[1/3,1/3,1/3,1/3,1/3,1/3,1,1/3,1/3,1/2,1,1/3,1/3,1/3,1,1,1/3,1/2]]
alpha = 0.9
gama = 0.8
shift_fix(P,v,alpha,gama,n)
