# --------------- These codes compute the multilinear pagerank problem ---------------------
#compute the kron of x
function compute_kron(index, x, m)
  n = size(x,1)
  index = index - 1
  ind_vec = zeros(m)
  count = 1
  while(count < m+1)
    if (index<n)
      ind_vec[count] = index
    else
      ind_vec[count] = index%n
    end
    count+=1
    index = div(index,n)
  end
  result = 1
  for i in 1:m
    result = result*x[ind_vec[i]+1]
  end
  return result
end

# row_vec, col_vec, val_vec is a sparse n^m by n matrix, x is a n-dim vector
function sparse_kron(row_vec, col_vec, val_vec, x, m)
  n = size(x)[1];Px = zeros(n)
  col = col_vec[1];sumval = 0
  #println("debug")
  for ind = 1:size(val_vec)[1]
    if col_vec[ind]!= col
      Px[col]=sumval
      sumval = val_vec[ind]*compute_kron(row_vec[ind],x,m)
      col = col_vec[ind]
    else
      sumval += val_vec[ind]*compute_kron(row_vec[ind],x,m)
    end
    if ind==size(val_vec)[1]
      Px[col]=sumval
    end
  end
  return Px
end


# P is a sparse n^m by n matrix
function shift_fix(P, v, alpha, gama, n, m)
  maxiter = 10000
  tol = 1/10^(8)
  x_old = rand(n)
  x_old = x_old/sum(x_old)
  row_vec, col_vec, val_vec = findnz(P)
  for i = 1:maxiter
    Px = sparse_kron(row_vec, col_vec, val_vec, x_old, m)
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
n=3;m=2
P = sparse([1/3 1/3 1/3; 1/3 1/3 1/3; 1/3 1/3 1/3; 1/3 1/3 1/3; 0 0 1; 0 1/2 1/2; 0 1 0;0 0 1; 0 1 0])
v = ones(3,1)/3
alpha = 0.9
gama = 0.8
shift_fix(P,v,alpha,gama,n,m)
