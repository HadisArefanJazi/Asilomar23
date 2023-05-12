%% result of 1000 initial conditions for four lambda (1, 0.1, 0.01, 0.001)
close all;
clear;
clc;
Rx = [5.4825    2.5050    1.1184    2.7127
      2.5050    5.0025    2.1642    1.2115
      1.1184    2.1642    5.5001    1.6464
      2.7127    1.2115    1.6464    6.5783];
Rx_inx            = inv(Rx);
Parameters.a      = 2;
Parameters.M      = 4;
Parameters.Rx     = Rx;
Parameters.ConNum = 1;
x0 = [1 0 0 1 1 0 0 1]';

alpha             = 0.0001; %learning rate
tol               = 0.0000001;
max_iter          = 40000;
Lambda            = [0.3 ,0.2];
for i=1:length(Lambda)
    Parameters.lambda = Lambda(i);
    [x_opt, history]  = gradient_descent(x0, alpha, max_iter, tol,Parameters);
  for j=1:1000
      x_optbefore              = -5 + 10*rand(8,1);
      [x_opt, cost, mse, attention]  = gradient_descent(x_optbefore, alpha, max_iter, tol,Parameters);
     
    Mycost(j,i)                      = cost;
    Myattention(j,i)                 = attention;
    x_optbefore(:,j,i)               = x_optbefore;
    Mymse(j,i)                       = mse; 
  end
end
[Rn_inv, ~, ~]    = MatrixCreator(x_opt,Parameters);
MSE               = trace(inv(Rx_inx + Rn_inv))
attention         = trace(Rn_inv)
save ('initials-new.mat',"Mycost","Lambda","Myattention","Mymse");

%--------------------------------------------------------------------------
function [x_opt, mycost, mymse, myattention] = gradient_descent(x0, alpha, max_iter, tol,Parameters)
x = x0;
for i = 1:max_iter
    df = gradient(x,Parameters);
    x = x - alpha* df;
    [c,mse,attention] = objective(x,Parameters);
    cost(i)=c;
    if i>2
        if  (cost(i-1)-cost(i)) < tol
            break;
        end
    end
    [cost(i),mse(i),attention(i)] = objective(x,Parameters);
    mycost                        = cost(i);
    mymse                         = mse(i);
    myattention                   = attention(i);
end

    x_opt = x;
end
function df = gradient(X,Parameters)
ConNum      = Parameters.ConNum;
init_V1_V2  = X(ConNum:end);
a           = Parameters.a; %size of N1
m           = Parameters.M; % size of Rn
Rx          = Parameters.Rx;
lambda      = Parameters.lambda;
Rx_inv      = eye(size(Rx))/Rx;
X1          = Rx_inv(1:a,1:a);
X2          = Rx_inv(1:a,a+1:m);
X3          = Rx_inv(a+1:m,a+1:m);
[~, V1, V2] = MatrixCreator(init_V1_V2,Parameters);
A           = X1 + V1*V1';
B           = X3 + V2*V2';
M           = A - X2/B*X2';
K           = B - X2'/A*X2 ;
KINV        = eye(size(K))/K;
MINV        = eye(size(M))/M;
AINV        = eye(size(A))/A;
BINV        = eye(size(B))/B;
F           = AINV'*X2*KINV'*KINV'*X2'*AINV';
Z           = BINV'*X2'*MINV'*MINV'*X2*BINV'; 
term1       = -((M^(-2))'+ M^(-2))*V1 - (F'+F) *V1 + 2*lambda*V1; %Stationarity Condition
term2       = -((K^(-2))'+ K^(-2))*V2 - (Z'+Z) *V2 + 2*lambda*V2; %Stationarity Condition
df1         = reshape (term1,a*a,1);
df2         = reshape (term2,(m-a)*(m-a),1);
df          = [df1;df2];
end
function [cost, mse, attention]  = objective(X,Parameters)
ConNum      = Parameters.ConNum;
init_V1_V2  = X(ConNum:end);
a           = Parameters.a; %size of R1
m           = Parameters.M; % size of R
Rx          = Parameters.Rx;
lambda      = Parameters.lambda;
Rx_inv      = eye(size(Rx))/Rx;
X1          = Rx_inv(1:a,1:a);
X2          = Rx_inv(1:a,a+1:m);
X3          = Rx_inv(a+1:m,a+1:m);
[Rn_inv, V1, V2] = MatrixCreator(init_V1_V2,Parameters);
A           = X1 + V1*V1';
B           = X3 + V2*V2';
AINV        = eye(size(A))/A;
BINV        = eye(size(B))/B;
cost        = trace((A - X2*BINV*X2')^(-1)) +...
              trace((B - X2'*AINV*X2)^(-1)) +...
              lambda*(trace(Rn_inv));
mse=         trace((A - X2*BINV*X2')^(-1))+trace((B - X2'*AINV*X2)^(-1));
attention   = trace(Rn_inv);
end
function [Rn_inv, V1, V2] = MatrixCreator(v,Parameters)
a           = Parameters.a; %size of R1
m           = Parameters.M; % size of R
v1          = v(1:a*a);
v2          = v(a*a+1:end);
V1          = reshape(v1,a,a);
V2          = reshape(v2,m-a,m-a);
Rn_inv      = [V1*V1' zeros(a,m-a);zeros(m-a,a) V2*V2'];
end