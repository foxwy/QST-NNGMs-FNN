function rho = Rand_DM(N, purity)
    x = 1:2^N; %eigenvalue index
    lambda = 0;
    purityTemp = 0;
    %Generate exponialy decreasing eigenvalues with the specified purity
    while purityTemp < purity
        lambda = lambda + 0.001; %increase std until reach correct purity
        lam = exp(-lambda * x); %exponential distribution of eigenvalues
        lamb = lam / sum(lam);
        purityTemp = sum(lamb.^2);
    end

    %Generate completely random density matrix with predefined eigenvalues
    rho = makeRandomDensityMatrix(lamb);
end

%%
function [rho] = makeRandomDensityMatrix(lambda)
  %generate random density matrix
    d = length(lambda);
    rho = zeros(d);
    randM = rand(d).*exp(1i*2*pi*rand(d));

    [Q, R] = qr(randM);
    
    for ii=1:d
        psi = Q(:,ii);
        rho = rho + psi*psi'*lambda(ii);
    end
end