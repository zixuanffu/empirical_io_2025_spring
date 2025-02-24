function [profit] = ccprofit(nfirms,descn,binom,D,f,ggamma)
% For a given number of active firms, computes the profit and market share 
% for the static Cournot competition for each firm in all  states. 

% initialize function output
profit = zeros(descn,nfirms); % Profit for each state-firm

for i = 1:descn
    if mod(i, 50) == 0 % report progress
        fprintf('  Computed: %d\n', i);
    end

    w = decode(i,nfirms,binom); % decode state i to state w
    theta = ggamma * exp(-(w-4));  % marginal cost

    %   Solve for cournot equilibrium with n firms; reduce n until all firms
    %   want to produce quantity > 0
    n=nfirms;
    p = (D + sum(theta(1:n)))/(n+1); % eqm price for n firms
    while ~((p - theta(n) >= 0) || (n==1)) % reduce n until either positive q or 1 firm left 
        n=n-1;
        p = (D + sum(theta(1:n)))/(n+1);
    end
    q = zeros(nfirms,1);
    if p - theta(n) > 0
        q(1:n) = p - theta(1:n); % eqm quantity
    end
    quan=q;

    pstar = D - sum(quan);   % Equilibrium price
    profstar = (pstar>theta).*(pstar-theta).*quan - f; % Equilibrium profits

    profit(i,:) = profstar'; 
end