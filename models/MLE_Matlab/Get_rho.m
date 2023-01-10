function [ket, rho] = Get_rho(type, N, P)
    ket_0 = [1; 0];
    ket_1 = [0; 1];
    
    if type == 'G'
        ket_00 = ket_0;
        ket_11 = ket_1;
        for i=1:N-1
            ket_00 = kron(ket_00, ket_0);
            ket_11 = kron(ket_11, ket_1);
        end
        ket_ghz = 1/sqrt(2)*(ket_00 + ket_11);
        ket = ket_ghz;
        rho = P*(ket_ghz*ket_ghz') + (1-P)*eye(2^N)/2^N;
        
    elseif type == 'I'
        ket_00 = ket_0;
        ket_11 = ket_1;
        for i=1:N-1
            ket_00 = kron(ket_00, ket_0);
            ket_11 = kron(ket_11, ket_1);
        end
        ket_ghzi = 1/sqrt(2)*(ket_00 + 1j*ket_11);
        ket = ket_ghzi;
        rho = P*(ket_ghzi*ket_ghzi') + (1-P)*eye(2^N)/2^N;
        
    elseif type == 'W'
        ket_w = zeros(2^N,1);
        ket_w(2.^(0:N-1)+1) = 1/sqrt(N);
        ket = ket_w;
        rho = P*(ket_w*ket_w') + (1-P)*eye(2^N)/2^N;
    
    elseif type == 'P'
        ket01 = 1/sqrt(2)*(ket_0 + ket_1);
        ket_p = ket01;
        for i=1:N-1
            ket_p = kron(ket_p, ket01);
        end
        ket = ket_p;
        rho = P*(ket_p*ket_p') + (1-P)*eye(2^N)/2^N;
        
    elseif type == 'R'
        ket = 1;
        rho = Rand_DM(N, P);
  
    end
end