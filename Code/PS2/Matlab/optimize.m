function [nx_t,nval_t] = optimize(w,...
    oldvalue,oldx,isentry,profit,...
    dtable,etable,multfac,two_n,kmax,nfirms,mask,...
    phi,entry_k,beta,delta,a)
% This procedure calculates optimal investment and value function for a
% given state w. 

% Main input vars:
% w = state code (integer)

% Output vars:
% nx_t = new investment strategy (row vector, 1 X nfirms)
% nval_t = new value function (row vector, 1 X nfirms)

locw = qdecode(w,dtable); % state tuple
locwx = locw; % state tuple after exited firms have exited
oval = oldvalue(w,:)'; % old value function at state w
ox = oldx(w,:)'; % old investment at state w
nval = zeros(nfirms,1); % new value column vector
nx = zeros(nfirms,1); % new investment column vector

%% Exit firms: 
% find which firms exit and update locwx to reflect the exits
%%%%%%%% Task %%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%
%% Entry probability:
% compute the probability of a potential entrant entering
% initialize new variable locwe to be equal to locwx and update locwe 
% to reflect the possible future state if the potential entrant enters
%%%%%%%% Task %%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%
%% Calculate the optimal policies for this industry structure, given that
% entry and exit are as specified.
for j = 1:nfirms
    % if firm j is in state 0, firm j and all firms with lower efficiency
    % level exit and gets scrap value
    % if firm j is in state > 0 (active firms), compute value of investing and
    % not investing (make sure to take expectation over entry
    % possibilities)
    %%%%%%% Task %%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%
    %% Update investment for firm j:
    % solve for the optimal probability of investment for firm j based on 
    % updated value of investing and not investing (you should be able to 
    % write down an analytical solution for optimal investment)
    %%%%%%% Task %%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%
    %% Update value for firm j given investment:
    % update the value function for firm j
    % exit decisions for firm j might change given the new value. if so,
    % update value and investment decision of firm j.
    % update exit deicision all firms with lower efficiency level if needed
    % update ox, locwx, locwe, so that the rest if the firms in the loop
    % take the new policies into account.
    %%%%%%% Task %%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%
end
nx_t = nx';
nval_t = nval';
end