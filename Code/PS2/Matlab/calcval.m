function [val_up,val_stay] = calcval(place,w,x,k,...
    oldvalue,...
    etable,multfac,two_n,kmax,nfirms,mask,...
    delta,a)
% This procedure calculates the continuation value for increasing the 
% efficiency level (i.e., moving up) and staying at efficiency level for a 
% given state w.

% Main input vars:
% place = place of the firm in w (integer, j in the problem set)
% w = state tuple (nfirms X 1)
% x = investment strategy in state w (nfirms X 1)
% k = own state (integer) 

% Output vars:
% val_up = value of moving up in efficiency level (integer)
% val_stay = value of staying at the same efficiency level (integer)

z1 = zeros(nfirms,1); % Lower bound for states
z2 = kmax*ones(nfirms,1); % Upper bound for states

% Adjust ``mask'' based on the firm's place such that the firm stays
if nfirms > 1
    if place == 1
        locmask = [zeros(1,two_n);mask];
    elseif place == nfirms
        locmask = [mask;zeros(1,two_n)];
    else 
        locmask = [mask(1:place-1,:);zeros(1,two_n);mask(place:nfirms-1,:)];
    end
else
    locmask = zeros(1,1);
end

% Modify investment and state
x(place) = 0; % Set own investment to zero
w(place) = k; % Set own state to k
justone = zeros(nfirms,1); % dummy vector for "place", used to update firm's investment outcome
justone(place) = 1;

% Probability of moving up
p_up = (a .* x) ./ (1 + a .* x);

% Initialize output
val_up=0; 
val_stay=0;

for i = 1:two_n
    % Compute transition probability (the probability of each of the
    % two_n outcomes)
    probmask = prod((locmask(:, i) .* p_up) + ((1 - locmask(:, i)) .* (1 - p_up)));
    %% Value for k_v, i.e., firm in place does not move up
    % Compute new state d
    d = w+locmask(:,i); % private shock
    temp = sortrows([d,justone],1,'descend');
    d = temp(:,1);
    e = d - 1; % aggregate shock

    % Check for evaluation of value fn. at -1
    e = max([e,z1],[],2);
    % Check for evaluation of value fn. at kmax+1
    d = min([d,z2],[],2);
    pl1 = maxind(temp(:,2)); % which firm is "place" in the new state

    % update expected value
    val_stay = val_stay + ((1-delta)*oldvalue(qencode(d,etable,multfac),pl1) ...
        + delta*oldvalue(qencode(e,etable,multfac),pl1))*probmask;
    %% Value for k_v+1, i.e., firm in place moves up
    %%%%%%%% Task %%%%%%%%%
   
    %%%%%%%%%%%%%%%%%%%%%%
end
end