function shift_fix(P, v, alpha, gama, n, m)
  maxiter = 10000
  tol = 1/10^(8)
  x_old = rand(n)
  x_old = x_old/sum(x_old)
  println("sfdadf")
  for i = 1:maxiter
    xm = x_old
    for j = 1:m-1
      xm = kron(xm, x_old)
    end
    x_new = (alpha/(1+gama))*P*xm + ((1-alpha)/(1+gama))*v + (gama/(1+gama))*x_old
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

n=3;m=2
P = transpose([1/3 1/3 1/3; 1/3 1/3 1/3; 1/3 1/3 1/3; 1/3 1/3 1/3; 0 0 1; 0 1/2 1/2; 0 1 0;0 0 1; 0 1 0])
v = ones(3,1)/3
alpha = 0.9
gama = 0.8
shift_fix(P,v,alpha,gama,n,m)
# test git second commit
