function [] = static_profit(c)
%  loops the number of active firms from 1 to N to compute static profit and market share
nfmax = c.MAX_FIRMS; % max number of active firms
kkmax = c.KMAX;      % max efficiency level 

% Set up binomial coefficients for decoding/encoding of n-tuples
kmax = kkmax;
binom = eye(nfmax+kmax+2);
binom = [zeros(nfmax+kmax+2,1),binom];
for i = 2:nfmax+kmax+2 % row i+1, column 2:(i+2) captures i choose 0 to i choose i
    binom(i,2:i) = binom(i-1,2:i) + binom(i-1,1:i-1);
end

for nfirms = 1:nfmax
    % Number of descending n-tuples
    fprintf('\nFirms: %d\n', nfirms)
    descn = binom(nfirms+kmax+1,kmax+2); % nfirms in kmax+1 states
    % disp(nchoosek(nfirms+kmax, kmax)), nfirms in kmax+1 states
    fprintf('Industry structures to compute: %d\n', descn)

    D = c.INTERCEPT; % cournot demand intercept 
    f = c.FIXED_COST; % cournot fixed cost
    ggamma = c.GAMMA; % cournot marginal cost coefficient
    [profit] = ccprofit(nfirms,descn,binom,D,f,ggamma);

    % write output
    s = int2str(nfirms);
    save(['a.' c.PREFIX '_pr' s '.mat'], 'profit');
end
c.PROFIT_DONE=1;

end


