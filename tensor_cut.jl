
# --------------- These codes compute the tensor spectral clustering problem ---------------------
# --------------- The paper can be found at :http://arxiv.org/abs/1502.05058

# --------------- Author: Tao Wu
#---------------- Email: wutao27@gmail.com

# compute transition matrix from the tensor P and the distribution x
function tran_matrix(P, x)
  n = int64(maximum(P[1]))  #transition matrix column length
  RT = zeros(n,n)
  m = size(P,1)-1  # tensor dimension
  NZ = size(P[1],1) # total number of non-zero items in the tensor
  for i = 1:NZ
    ind1 = P[1][i];ind2 = P[2][i];val = P[m+1][i]
    mul = 1
    for j = 3:m
      mul = mul * x[P[j][i]]
    end
    itemVal = val * mul
    RT[ind2,ind1] += itemVal
  end
  return sparse(RT)
end

# sweep_cut function, where permEv is the permutation of sorted egenvectors
function sweep_cut(T, permEv)
  n = length(permEv) #length of egenvector
  m = size(T,1)-1  # tensor dimension
  TN = sum(T[m+1]) # sum of values in the tensor
  per = sortperm(permEv)
  cutS1 = zeros(n+1);cutS2 = zeros(n+1);volS1 = zeros(n+1)
  cut = zeros(n-1);cutS2[1]=TN;

  message(FILE_RUNTIME,"parse every entry in the tensor")
  for ind = 1:size(T[1],1)
    val = T[m+1][ind]
    rk = [per[ T[j][ind] ] for j = 1:m]
    r1 = minimum(rk);r2 = maximum(rk)
    cutS1[r2+1] += val
    cutS2[r1+1] += val
    volS1[ per[ T[1][ind] ] + 1] += val
  end

  message(FILE_RUNTIME,"calculate cuts")
  for i=1:n-1
    cutS1[i+1] += cutS1[i];
    cutS2[i+1] = cutS2[i] - cutS2[i+1];
    volS1[i+1] += volS1[i];

    cut[i] = (TN - cutS1[i+1] - cutS2[i+1])/(3*min(volS1[i+1],TN-volS1[i+1]))
  end
  cutPoint = indmin(cut)
  return (cutPoint, cut, cut[cutPoint], [ TN, cutS1[cutPoint+1], cutS2[cutPoint+1], volS1[cutPoint+1] ])
end
