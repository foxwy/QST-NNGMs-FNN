%% Born-rule probabilities of a single-qubit state
% Let us first write down the identity and Pauli matrices.
identity = [ 1   0 
             0   1 ];
pauli_x =  [ 0   1
             1   0 ];
pauli_y =  [ 0 -1i 
            1i   0 ];
pauli_z =  [ 1   0 
             0  -1 ];
         
%%
% However, instead of using tetrahedron measurements, let us use Pauli
% measurements on each qubit.
tetrahedron = zeros(2,2,4);
tetrahedron(:,:,1) = 1/4*(identity+( pauli_x+pauli_y+pauli_z)/sqrt(3));
tetrahedron(:,:,2) = 1/4*(identity+(-pauli_x-pauli_y+pauli_z)/sqrt(3));
tetrahedron(:,:,3) = 1/4*(identity+(-pauli_x+pauli_y-pauli_z)/sqrt(3));
tetrahedron(:,:,4) = 1/4*(identity+( pauli_x-pauli_y-pauli_z)/sqrt(3));

pauli = zeros(2,2,6);
pauli(:,:,1) = (identity+pauli_x)/6;
pauli(:,:,2) = (identity-pauli_x)/6;
pauli(:,:,3) = (identity+pauli_y)/6;
pauli(:,:,4) = (identity-pauli_y)/6;
pauli(:,:,5) = (identity+pauli_z)/6;
pauli(:,:,6) = (identity-pauli_z)/6;

%load 'rho.mat'
%load 'measurements.mat'

%% Ex: 2 (Convergence Experiment of Random Mixed States for Different Qubits, no noise)
% savedir = 'result/';
% State = ['W' 'I' 'P' 'R'];
% N_qubits = [2 3 4 5 6 7 8 9];
% for j=1:numel(State)
%     State(j)
%     r_path = [savedir State(j) '/'];
%     if ~exist(r_path, 'dir')
%         mkdir(r_path)
%     end
%     
%     for N_i=1:numel(N_qubits)
%         n_qubit = N_qubits(N_i)
%         save_data = struct;
%         P = rand(1, 50);
%         
%         for ii=1:numel(P)
%             ii
%             [ket, rho] = Get_rho(State(j), n_qubit, P(ii));
%             %figure(1);
%             %show_matrix(rho);
% 
%             % measurements
%             % Now let us compute the Born-probabilities.
%             basis = repmat({tetrahedron},[1 n_qubit]);
%             probs = qmt(rho, basis);
%             %display(probs)
% 
%             % reconstruction
%             % Now let's find the maximum-likelihood estimator (and keep some statistics
%             % on the time taken!).
%             opts = struct;
%             opts.imax = 500;
%             opts.rho_star = rho;
%             
%             %% CG-APG
%             %[rho_qb, stats_qb_mle] = qse_apg(basis, probs, opts);
%             
%             %% iMLE
%             [rho_qb, stats_qb_mle] = iMLE(basis, probs, opts);
%             
%             if ii == 1
%                 save_data.P = P(ii);
%                 save_data.stats = stats_qb_mle;
%             else
%                 save_data(ii).P = P(ii);
%                 save_data(ii).stats = stats_qb_mle;
%             end
%             %figure(2);
%             %show_matrix(rho_qb);
%         end
%         
%         savePath = [r_path 'iMLE_' num2str(n_qubit) '.mat'];
%         save(savePath, 'save_data');
%     end
% end 

%% Ex: 3 (Convergence Experiment of Random Mixed States for Different Samples)  random_P
% n_qubit = 8;
% savedir = 'result/';
% State = ['R'];
% N_s = [100 500 1000 5000 10000 50000 100000 1000000];
% for j=1:numel(State)
%     State(j)
%     r_path = [savedir State(j) '/'];
%     if ~exist(r_path, 'dir')
%         mkdir(r_path)
%     end
%     
%     for N_i=1:numel(N_s)
%         N_s(N_i)
%         save_data = struct;
%         P = rand(1, 50);
%         
%         for ii=1:numel(P)
%             ii
%             [ket, rho] = Get_rho(State(j), n_qubit, P(ii));
%             %figure(1);
%             %show_matrix(rho);
% 
%             % measurements
%             % Now let us compute the Born-probabilities.
%             basis = repmat({tetrahedron},[1 n_qubit]);
%             probs = qmt(rho, basis);
%             %display(probs)
% 
%             % measurement
%             N = N_s(N_i); 
%             counts = histc(rand(N,1), [0; cumsum(probs)]);
%             counts = counts(1:end-1);
%             mea_p = counts / N;
% 
%             % reconstruction
%             % Now let's find the maximum-likelihood estimator (and keep some statistics
%             % on the time taken!).
%             opts = struct;
%             opts.imax = 500;
%             opts.rho_star = rho;
%             
%             %% CG-APG
%             %[rho_qb, stats_qb_mle] = qse_apg(basis, mea_p, opts);
%             
%             %% iMLE
%             [rho_qb, stats_qb_mle] = iMLE(basis, mea_p, opts);
%             
%             if ii == 1
%                 save_data.P = P(ii);
%                 save_data.stats = stats_qb_mle;
%             else
%                 save_data(ii).P = P(ii);
%                 save_data(ii).stats = stats_qb_mle;
%             end
%             %figure(2);
%             %show_matrix(rho_qb);
%         end
%         
%         savePath = [r_path 'iMLE_S' num2str(N_s(N_i)/100) '.mat'];
%         save(savePath, 'save_data');
%     end
% end 

%% Ex: 4 (Convergence Experiment of Pure States with Depolarizing Noise)
% n_qubit = 8;
% savedir = 'result/';
% State = ['W']% 'P' 'I'];
% N_s = [10000];
% for j=1:numel(State)
%     State(j)
%     r_path = [savedir State(j) '/'];
%     if ~exist(r_path, 'dir')
%         mkdir(r_path)
%     end
%     
%     for N_i=1:numel(N_s)
%         N_s(N_i)
%         save_data = struct;
%         P = rand(1, 1);
%         
%         for ii=1:numel(P)
%             ii
%             [ket, rho] = Get_rho(State(j), n_qubit, 1-P(ii));
%             %figure(1);
%             %show_matrix(rho);
% 
%             % measurements
%             % Now let us compute the Born-probabilities.
%             basis = repmat({tetrahedron},[1 n_qubit]);
%             probs = qmt(rho, basis);
%             %display(probs)
% 
%             % measurement
%             N = N_s(N_i); 
%             counts = histc(rand(N,1), [0; cumsum(probs)]);
%             counts = counts(1:end-1);
%             mea_p = counts / N;
% 
%             % reconstruction
%             % Now let's find the maximum-likelihood estimator (and keep some statistics
%             % on the time taken!).
%             opts = struct;
%             opts.imax = 500;
%             [ket, rhoo] = Get_rho(State(j), n_qubit, 1);
%             opts.rho_star = rhoo;
%             opts.ket = ket;
%             
%             %% CG-APG
%             [rho_qb, stats_qb_mle] = qse_apg(basis, mea_p, opts);
%             
%             %% iMLE
%             %[rho_qb, stats_qb_mle] = iMLE(basis, mea_p, opts);
%             
%             if ii == 1
%                 save_data.P = P(ii);
%                 save_data.stats = stats_qb_mle;
%             else
%                 save_data(ii).P = P(ii);
%                 save_data(ii).stats = stats_qb_mle;
%             end
%             %figure(2);
%             %show_matrix(rho_qb);
%         end
%         
%         %savePath = [r_path 'iMLE_S_N' num2str(N_s(N_i)/100) '.mat'];
%         %save(savePath, 'save_data');
%     end
% end 

%% test
State = 'I';
n_qubit = 5;
[ket, rho] = Get_rho(State, n_qubit, 1);
basis = repmat({tetrahedron},[1 n_qubit]);
probs = qmt(rho, basis);

opts = struct;
opts.imax = 1000;
opts.rho_star = rho;
opts.ket = ket;

%% CG-APG
[rho_qb, stats_qb_mle] = qse_apg(basis, probs, opts);

%% iMLE
%[rho_qb, stats_qb_mle] = iMLE(basis, probs, opts);


%% figure
% figure(3);
% iterations = 1:numel(Fq);
% semilogx(iterations, Fq, iterations, Td, 'black', 'linewidth', 2)
% xlabel('Iterations')
% ylabel('Fidelity, Trace distance')
% ylim([0, 1])
% legend({'APG-MLE Fidelity', 'APG-MLE Trace distance'}, 'Location', 'northwest')
% grid on
% 
% %%
% % Here is a graph of the difference in log likelihood as the algorithm
% % progresses, using information from |stats_8qb_mle|.
% figure(4);
% %semilogy(stats_qb_mle.times,(stats_qb_mle.fvals-min(stats_qb_mle.fvals)));
% semilogy(stats_qb_mle.times, Fq)
% xlabel('time (s)');
% ylabel('difference in log likelihood from optimal');
% snapnow;

% load 'fidelities-apg-mle.mat'
% 
% %% figure
% iterations = 1:numel(flist1);
% semilogx(iterations, flist1, 'black', 'linewidth', 2)
% xlabel('Iterations')
% ylabel('Fidelity')
% ylim([0, 1])
% legend('APG-MLE', 'Location', 'northwest')
% grid on