%-*-Text-*-
% file = gen_kSAT.m

function  M = gen_kSAT(k,n,m)
% return sparse (m x 2n) matrix M encoding a satisfiable
%    uniform random k-SAT instance
% (cf. https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html)
% n = #var, m = #clause
% time includes matrix generation time

     n2 = 2*n;
      M = spalloc(m,n2,k*m);  %<= sparse (k*m = max #nonzero)
   mdl0 = rand(n,1)<0.5;      % radomly generate a satisfying model
ans_mdl = [mdl0;1-mdl0];

for s=1:m                     % create m clauses satisfied by mdl0
   M(s,:) = 0;
   do
      x = randperm(n,k);      % random x=[x1..xk], 1=<xi=<n, no repetition
      y = (rand(1,k)<0.5);    % randomly decide the sign of k literals
      v = x(find(y==1));      % pos literals
      w = x(find(y==0));      % neg literals
      z = [v w+n];
   until( any(ans_mdl(z)) );  % guarantee satisfiability
   M(s,z) = 1;
endfor;

%printf(" (ans_mdl |= M) = %d\n",all(M*ans_mdl));
%if (all(M*ans_mdl)==0) M = -1; endif;

