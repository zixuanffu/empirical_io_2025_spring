function [] = eql_ma(c)
% This code computes the dynamic equilibrium
rlnfirms = c.MAX_FIRMS; % max number of active firms
kmax = c.KMAX; % max efficiency level
stfirm = c.START_FIRMS; % starting number of firms for equilibrium computation

x_entryl = c.ENTRY_LOW;
x_entryh = c.ENTRY_HIGH;
phi = c.SCRAP_VAL;
entry_k = c.ENTRY_AT;
beta = c.BETA;
delta = c.DELTA;
a = c.INV_MULT;

tol = c.TOL;  % Tolerance for convergence
newvalue = []; newx = []; oldvalue = []; oldx = []; isentry = [];

% Set up binomial coefficients for decoding/encoding of n-tuples
binom = eye(rlnfirms+kmax+1);
binom = [zeros(rlnfirms+kmax+1,1),binom];
for i = 2:rlnfirms+kmax+1
    binom(i,2:i) = binom(i-1,2:i) + binom(i-1,1:i-1);
end

% solve for the game with rlnfirms in the market; start with a small number
% of firms (stfirm) and use the solution to initialize larger numbers
for nfirms = stfirm:rlnfirms
    % Number of different combinations of states
    wmax = binom(nfirms+kmax+1,kmax+2); % nfirms in kmax+1 states    
 
    % Load profit
    fprintf('\nFirms: %d   States: %d\nInitialization ...\n', nfirms, wmax);
    load(['a.' c.PREFIX '_pr' int2str(nfirms) '.mat'], 'profit')

    % Number of rivals' actions
    two_n = 2^(nfirms-1);

    % Matrix of all possible binary outcomes of rivals
    % each row is a rival, each column is an outcome
    % i.e., all binary numbers from 0 to two_n - 1
    mask = (dec2bin(0:2^(nfirms-1)-1) - '0')';

    %% Quick decoding table: map state id to state tuple (after symmetry and
    % anonymity)
    % Output: dtable
    dtable = zeros(nfirms,wmax);
    for i =1:wmax
        dtable(:,i) = decode(i,nfirms,binom);
    end

    %% Quick encoding table: map full industry state wgrid (before symmetry
    % and anonymity) to state id
    % Output: multfac, etable
    multfac = (kmax+1).^((1:nfirms)'-1); % this allows mapping without sorting
    % Create a cell array to hold grid matrices
    wgrid = cell(1, nfirms);
    [wgrid{:}] = ndgrid(0:kmax);
    % Convert cell to matrix and sort as descending
    wtable = zeros((kmax+1)^nfirms,nfirms);
    for i = 1:nfirms
        wtable(:, i) = wgrid{i}(:);
    end
    wtable = sort(wtable, 2, 'descend');
    clear wgrid
    % Create encoding table
    etable = zeros((kmax+1)^nfirms,1);
    for i = 1:size(wtable,1)
        etable(i) = encode(wtable(i,:)',nfirms,binom);
    end
    
    %% Initialize value and policy
    % Use value from nfirms-1 game, or for nfirms = 1, pick a number.
    [oldvalue,oldx] = initialize(dtable,nfirms,wmax,binom,newvalue,newx);

    %% Iterations
    fprintf('Contraction ...\n');
    ix = 1;
    norm = tol + 1;
    avgnorm = norm;
    while (norm > tol) && (avgnorm > 0.0001*tol)
        [newvalue,newx,isentry] = contract(...
            oldvalue,oldx,profit,...
            dtable,etable,multfac,wmax,two_n,kmax,nfirms,mask,...
            x_entryl,x_entryh,phi,entry_k,beta,delta,a);
        norm = max(max(abs(oldvalue - newvalue)));
        avgnorm = mean(mean(abs(oldvalue - newvalue)));

        fprintf('  %2d    Sup norm: %8.4f      Mean norm: %8.4f\n', ...
            ix, norm, avgnorm);
        ix = ix+1;

        % aaaaa = abs(oldvalue-newvalue);
        % normind1 = maxind(max(aaaaa,[],2));
        % normind2 = maxind(max(aaaaa));
        % normcode = (qdecode(normind1,dtable))';
        % disp(["Max. elt is: " normind2 "," normcode "; Old value: "])
        % disp([oldvalue(normind1,normind2) "; New value: "])
        % disp(newvalue(normind1,normind2))

        oldx = newx; 
        oldvalue = newvalue;
    end

    %% Add warning if there is any investment at the highest level.
    w=kmax;
    if nfirms > 1
        w = [w;zeros(nfirms-1,1)];
    end
    if max(newx(qencode(w,etable,multfac):wmax,1)) > 0
        disp('Warning: Positive investment recorded at highest efficiency level.')
        disp('Please consider increasing the maximum efficiency level (kmax).')
    end

    %% Save data in file for summary stats and simulation
    prising = a.*newx./(1+a.*newx);
    save(['a.' c.PREFIX '_markov' int2str(nfirms) '.mat'], ...
        'newvalue', 'newx', 'prising', 'isentry')

end

c.EQL_DONE = 1;

end














