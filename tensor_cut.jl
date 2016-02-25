# --------------- These codes compute sweep_cut for biased conductance ---------------------

# --------------- Author: Tao Wu
#---------------- Email: wutao27@gmail.com

# compute transition matrix from the tensor P and the distribution x
function tran_matrix(P, x)
  n = Int64(maximum(P[1]))  #transition matrix column length
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
function sweep_cut(Px, x, permEv)
  n = length(permEv) #length of egenvector
  per = sortperm(permEv)
  tranS1 = zeros(n+1);tranS2 = zeros(n+1);volS1 = zeros(n+1)
  PS1 = zeros(n+1);PS2 = zeros(n+1)
  cut = zeros(n-1);

  tPx = transpose(Px)
  v = ones(n);
  v = Px*v;

  #tranAdd1 = 0; tranAdd2 = (transpose(v)*x)[1]

  #tranS2[1] = 1-tranAdd2; PS2[1] = 1;

  println("\tcalculate cuts")
  for i=1:n-1
    ind = permEv[i]
    tempRow = tPx[:,ind]; tempCol = Px[:,ind]
    tempRowInd = rowvals(tempRow)
    tempColInd = rowvals(tempCol)

    tranS1[i+1] = tranS1[i]; tranS2[i+1] = tranS2[i]

    for j in tempRowInd
      if per[j] > i
        tranS1[i+1] += x[ind]*tempRow[j]/v[ind]
      end
      if per[j] < i
        tranS2[i+1] = tranS2[i+1] - x[ind]*tempRow[j]/v[ind]
      end
    end

    for j in tempColInd
      if per[j] > i
        tranS2[i+1] += x[j]*tempCol[j]/v[j]
      end
      if per[j] < i
        tranS1[i+1] = tranS1[i+1] - x[j]*tempCol[j]/v[j]
      end
    end

    #tranAdd1 += x[ind]*v[ind]
    #PS1[i+1] = tranS1[i+1] + (n-i)*tranAdd1/n
    PS1[i+1] = tranS1[i+1]
    volS1[i+1] = volS1[i] + x[ind]
    prob1 = PS1[i+1]/volS1[i+1]
    # if prob1 > 1
    #   println("i = ($i) prob1 ($prob1) > 1")
    #   println("tran = ($(tranS1[i+1])) PS1 = ($(PS1[i+1])) volS1 = ($(volS1[i+1]))")
    #   error("error")
    # end

    #tranAdd2 = tranAdd2 - x[ind]*v[ind]
    #PS2[i+1] = tranS2[i+1] + i*tranAdd2/n
    PS2[i+1] = tranS2[i+1]
    prob2 = PS2[i+1]/(1 - volS1[i+1])

    # if prob2 > 1
    #   println("i = ($i) prob2 ($prob2) > 1")
    #   println("tran = ($(tranS2[i+1])) PS2 = ($(PS2[i+1])) volS1 = ($(volS1[i+1]))")
    #   error("error")
    # end
    cut[i] = max(prob1,prob2)
  end
  cutPoint = indmin(cut)
  return (cutPoint, cut, cut[cutPoint], [ PS1[cutPoint+1]; PS2[cutPoint+1]; volS1[cutPoint+1] ])
end
