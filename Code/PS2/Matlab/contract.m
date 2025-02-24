function [newvalue,newx,isentry] = contract(...
    oldvalue,oldx,profit,...
    dtable,etable,multfac,wmax,two_n,kmax,nfirms,mask,...
    x_entryl,x_entryh,phi,entry_k,beta,delta,a)
% This code performs one iteration on policy and value function

%% Entry decision
isentry = zeros(wmax,1);
for w =1:wmax
    locw = qdecode(w,dtable);
    % if there is space for entry, calculate entry value
    if locw(nfirms) == 0
        [~,v1] = calcval(nfirms,locw,oldx(w,:)',entry_k,...
            oldvalue,...
            etable,multfac,two_n,kmax,nfirms,mask,...
            delta,a);
        val = beta * v1;
        isentry(w) = (val - x_entryl) / (x_entryh - x_entryl);
    end
end
% Vector of entry probability for any rivals's state w1,...,wn-1
% bounded between 0 and 1
isentry = min([isentry,ones(wmax,1)],[],2);
isentry = max([isentry,zeros(wmax,1)],[],2);


%% Investment decision and value function
newx = zeros(wmax,nfirms);
newvalue = zeros(wmax,nfirms);
for w = 1:wmax
    [newx(w,:), newvalue(w,:)] = optimize(w,...
        oldvalue,oldx,isentry,profit,...
        dtable,etable,multfac,two_n,kmax,nfirms,mask,...
        phi,entry_k,beta,delta,a);
end
end


