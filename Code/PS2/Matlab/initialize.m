function [oldvalue,oldx] = initialize(dtable,nfirms,wmax,binom,newvalue,newx)
% This procedure takes the solved newx, newvalue matrix for the nfirms - 1
% problem, and puts them into the nfirms matrices oldx, oldvalue, for use
% as starting values
oldx = zeros(wmax,nfirms);
oldvalue = zeros(wmax,nfirms);
if nfirms == 1
    oldvalue = 1+0.1*(1:wmax)'; % newvalue and newx don't exist for nfirms-1=0
else
    for w = 1:wmax
        tuple = qdecode(w,dtable);
        % Initialize by mapping the current w to the corresponding nfirms-1 
        % equilibrium
        % (ignore the last firm) 
        n = encode(tuple(1:nfirms-1),nfirms-1,binom);
        oldvalue(w,1:nfirms-1) = newvalue(n,1:nfirms-1);
        oldx(w,1:nfirms-1) = newx(n,1:nfirms-1);

        % Initialize the last firm by ignoring the second last firm, 
        % i.e., swap the last two firms and set the last firm to 0 state
        tuple(nfirms-1) = tuple(nfirms);
        tuple(nfirms) = 0;
        n = encode(tuple,nfirms,binom);
        oldvalue(w,nfirms) = oldvalue(n,nfirms-1);
        oldx(w,nfirms) =  oldx(n,nfirms-1);
    end
end
end