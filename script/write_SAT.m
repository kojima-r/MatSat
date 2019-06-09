%-*-Text-*-
% file = write_SAT.m

function  fid = write_SAT(M,File)
% write SAT instance M into File in DIMACS format
% File created if not existing

[m n2] = size(M);   % M(m,2*n)  m=#clause, n=#var
n = n2/2;

M1 = M(1:m,1:n);
M2 = M(1:m,n+1:2*n);
X = M1-M2;
nzero = (X!=0);

fid = fopen(File,"w");
if (fid == -1) break; endif;

fprintf(fid, "p cnf %d %d\n",n,m);
        % delimiter = " " used ONCE between data
        % > dlmread(File," ",0,0)
for cl = 1:m
   for v = find(nzero(cl,:))
        if (X(cl,v)>0)
            fprintf(fid,"%d ",v);
        else
            fprintf(fid,"-%d ",v);
        endif;
        fflush(fid);
   endfor;
   fprintf(fid,"0\n");
   fflush(fid);
endfor;

fclose(fid);
