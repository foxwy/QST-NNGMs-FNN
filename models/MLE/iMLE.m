function [rho, stats] = iMLE(operators, f, opts)

if ~isa(operators,'cell') % nonseparable
    operators = {operators};
end

dims = zeros(numel(operators),2);
for i=1:numel(operators)
    temp = size(operators{i});
    dims(i,1) = temp(1);
    dims(i,2) = temp(end);
end
d = prod(dims(:,1));
K = prod(dims(:,2));
assert(K==numel(f));

% process options
defaults = struct;
defaults.rho_star = [];
defaults.ket = [];
defaults.imax = 20000;
defaults.rho0 = [];
defaults.save_rhos = false;

if ~exist('opts','var')
    opts = defaults;
else
    % scan for invalid options
    names = fieldnames(opts);
    for i=1:numel(names)
        if ~isfield(defaults, names{i})
            error('iMLE:opts','unknown option %s',names{i});
        end
    end
    % populate defaults if not in opts
    names = fieldnames(defaults);
    for i=1:numel(names)
        if ~isfield(opts, names{i})
            opts.(names{i}) = defaults.(names{i});
        end
    end
end

if ~isempty(opts.rho0)
    [V,D] = eig(opts.rho0);
    temp = full(diag(sparse(sqrt(max(0,diag(D))))))*V';
    A = temp/norm(temp,'fro');
    rho = A'*A;
else
    A = eye(d)/sqrt(d);
    rho = eye(d)/d;
end

% line search stuff
stats = struct;
stats.fvals = zeros(opts.imax,1);
stats.dists = zeros(opts.imax,1); % lower bound or actual trace distance between current iterate and rho_star
stats.Fq = zeros(opts.imax, 1);
stats.times = zeros(opts.imax,1);

stats.best_rho = [];
stats.best_fval = Inf;

if opts.save_rhos
    stats.rhos = {};
end

start = tic;

for i=1:opts.imax
    probs = qmt(rho, operators);
    
    adj = f./probs;  % if probs(j) == 0, what to do?
    adj(f==0)=0;
    rmatrix = qmt(adj, operators, 'adjoint');
    
    rho = rmatrix * rho * rmatrix;
    rho = rho / real(trace(rho));

    stats.dists(i) = 0.5*norm(rho-opts.rho_star,'fro');
    %stats.Fq(i) = real(trace(sqrtm(sqrtm(opts.rho_star)*rho*sqrtm(opts.rho_star)))^2);
    stats.Fq(i) = real(opts.ket'*rho*opts.ket);
    
    fval = -f(f~=0)'*log(probs(f~=0));
    stats.fvals(i) = fval; 
    if stats.fvals(i) < stats.best_fval
        stats.best_fval = stats.fvals(i);
        stats.best_rho = rho;
    end
    
    if opts.save_rhos
        stats.rhos{i,1} = rho;
    end
    
    stats.times(i) = toc(start); 
end

end




