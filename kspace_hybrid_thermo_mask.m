function [theta,A,c,f,Ac,algp] = kspace_hybrid_thermo_mask(acqp,thetainit,bathinit,algp)

%|function kspace_hybrid_thermo_mask
%|
%| Inputs:
%|  acqp    Acquisition parameters structure containing (required):
%|              data        [Nk,Nc]       Nc complex k-space data vectors
%|              k           [Nk,Nd]       Nd k-space sample vectors (cycles/cm)
%|                       OR [Nkx,Nky,Nkz] logical Cartesian k-space sampling mask
%|              fov         1             Field of view (cm) (Non-Cartesian only)
%|              mask        [Nx,Ny,Nz]    Binary mask over the FOV (Non-Cartesian only)
%|              L           [Nx*Ny*Nz*Nc,Nl] Multibaseline image library
%|  thetainit   [Nx,Ny,Nz]  Initial temperature map (real, negative). Real part is
%|                          temperature-induced phase at TE. (optional)
%|  algp    Algorithm parameters structure containing (structure and each entry are optional):
%|              order       1             Polynomial order (default = 0)
%|              lam         [1 2]         l1 penalty weights for real and imaginary parts of m
%|                                        (default = 10^-6)
%|              beta        1             Roughness penalty weight for real 
%|                                        and imaginary parts of m (default = 0)
%|              maskthresh  1             Phase threshold to obtain a mask of nonzero temperature-induced phase shifts, 
%|                                        for second stage of algorithm. (radians; default = 0.01)
%|              dofigs      1             Display intermediate figures (default = 0)
%|              thiters     1             Number of CG iterations per theta update (default = 10)
%|              citers      1             Number of CG iterations per c update (default = 5)
%|              masknz      [Nx,Ny,Nz]    Mask of non-zero heating
%|                                        locations. This will cause
%|                                        the algorithm to skip the l1-regularized 
%|                                        stage and go straight to the masked/unregularized stage 
%|                                        (default = [])
%|              maskhybr    [Nx,Ny,Nz]    Mask of locations within which to
%|                                        run the temperature reconstruction
%|                                        algorithm. For locations where the 
%|                                        mask is zero, will run conjugate  
%|                                        gradient image reconstruction.  
%|                                        This is used to exclude water  
%|                                        bath signal from brain heating data. 
%|
%| Outputs:
%|  theta       [Nx,Ny,Nz]    Complex temperature map
%|  A           [Nx*Ny*Nz,Np] Polynomial matrix (may be masked)
%|  c           [Np,Nc]       Polynomial coeffs
%|  f           [Nx,Ny,Nz,Nc] Baseline estimate
%|  Ac          [Nx,Ny,Nc]    Polynomial phase estimate (embedded into original mask)
%|  algp        struct        Final algorithm parameters structure
%|
%| Copyright 2014-05-19, William A Grissom, Pooja Gaur, Vanderbilt University
%| change from v2: run CG in bath instead of NLCG

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define optional inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist('algp','var')
    algp = struct();
end
if ~isfield(algp,'order')
    algp.order = 0; % zeroth-order only (phase drift)
end
if ~isfield(algp,'lam')
    algp.lam = 10^-6; % very small value
end
if ~isfield(algp,'beta')
    algp.beta = -1; % turn off roughness penalty if beta not supplied
end
if ~isfield(algp,'maskthresh')
    algp.maskthresh = 0.01; % small phase threshold
end
if ~isfield(algp,'dofigs')
    algp.dofigs = 0;
end
if ~isfield(algp,'thiters')
    algp.thiters = 10; % theta iterations
end
if ~isfield(algp,'citers')
    algp.citers = 5; % c iterations
end
if ~isfield(algp,'masknz')
    algp.masknz = [];
else
    if ~isempty(algp.masknz)
        algp.masknz = algp.masknz(acqp.mask);
    end
end


disp('Performing k-space hybrid thermometry.');
if islogical(acqp.k)
    disp('k-space is logical array; using Gmri_cart.');
    acqp.fov = [];
else
    disp('k-space is double array; using Gmri');
end

Nc = size(acqp.data,2); % Number of rx coils

%%%%%%%%%%%%%%%%%%%%%%%%%
% Build objects
%%%%%%%%%%%%%%%%%%%%%%%%%

% build polynomial matrix
A = buildA(algp.maskbrain,algp.order);

% build system matrix
[Gbrain,Gbath,Gall] = buildG(Nc,acqp.k,acqp.fov,algp.maskbrain,algp.maskbath,acqp.mask);

% build penalty object
if algp.beta > 0
    R = Robject(algp.maskbrain,'order',2,'beta',algp.beta,'type_denom','matlab');
else
    R = [];
end

% Normalize data to get to stabilize regularization performance between datasets
dnorm = median(abs(acqp.data(:)))*sqrt(length(acqp.data(:)));
if dnorm ~= 0
  acqp.data = acqp.data / dnorm;
  acqp.L = acqp.L / dnorm;
else
  disp ['Warning: normalization = 0, so not applied. This can ' ...
        'happen when the object has been masked. lam ' ...
        'may need tweaking.'];
end


%%%%%%%%%%%%%%%%%%%%%%%%%
% get initial f,c,theta
%%%%%%%%%%%%%%%%%%%%%%%%%

% mask initial theta
thetainit = thetainit(algp.maskbrain);
theta = thetainit;

% force negativity
%theta(theta >= 0) = 0;

% initialize c to zero
c = zeros(size(A,2),1);
Ac = A*c; 

% get initial brain baseline estimate
fBrain = f_update_brain(acqp.data,Ac,theta,acqp.L(algp.maskbrain,:),Gbrain);
% initial bath baseline is zero
bathinit = bathinit(algp.maskbath(:));
fBath = bathinit;
%fBath = zeros(sum(algp.maskbath(:)),1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L1-Penalized Component
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(algp.masknz) % only run if we don't already have a heating mask 
    
    costOld = Inf;
    
    cost = cost_eval(acqp.data,Gbrain,fBrain,Ac,theta,thetainit,R,algp.lam,algp.gamma,algp,Gbath,fBath,bathinit);
    itr = 0;
    fprintf('L1-penalized iteration %d, cost = %f\n',itr,cost);
    while costOld-cost >= algp.stopThresh*costOld
                       
        % update baseline in water bath
        %if itr > 0
        %    algp.doBathCG = 1;
        %end
        dataBath = acqp.data(:) - Gbrain*(fBrain.*exp(1i*(Ac+theta))); % remove brain signal
        %fBath = f_update_bath(fBath,dataBath,acqp.L(algp.maskbath,:),acqp.kmask,Gall,algp);
        fBath = f_update_bath(dataBath,fBath,Gbath,bathinit,algp);
        
        % update baseline in brain
        dataBrain = acqp.data(:) - Gbath*fBath; % remove bath signal
        fBrain = f_update_brain(dataBrain,Ac,theta,acqp.L(algp.maskbrain,:),Gbrain);

        % update poly coeffs
        c = c_update(dataBrain,A,c,theta,fBrain,Gbrain,algp);
        Ac = A*c; 
                
        % update temp phase shift
        theta = theta_update(dataBrain,Ac,theta,thetainit,fBrain,Gbrain,algp,algp.lam,algp.gamma,R);
        
        if algp.dofigs;
            figure(201);
            subplot(231); imagesc(embed(-real(theta),algp.maskbrain).'); axis image; title 'Estimated phase';colorbar
            subplot(232); imagesc(embed(imag(theta),algp.maskbrain).'); axis image; title 'Estimated Nepers';colorbar
            subplot(233); imagesc(embed(-real(theta) >= algp.maskthresh,algp.maskbrain).'); axis image; title 'Significant phase'
            subplot(234); im(embed(fBath,algp.maskbath));
            drawnow;
        end
        
        % calculate cost with updated parameters
        costOld = cost;
        cost = cost_eval(acqp.data,Gbrain,fBrain,Ac,theta,thetainit,R,algp.lam,algp.gamma,algp,Gbath,fBath,bathinit);
        
        itr = itr + 1;
        fprintf('L1-penalized iteration %d, cost = %f\n',itr,cost);
        
    end
    
    % get a mask of potential temperature shifts.
    algp.masknz = -real(embed(theta,algp.maskbrain)) >= algp.maskthresh;
    %theta = theta.*algp.masknz; 
    
    theta(-real(theta) < algp.maskthresh) = 0;
    %algp.lam = [0 0];
    algp.lam = 0*algp.lam./100; % just reduce it for masked, to avoid points that blow up
    %algp.gamma = 0*algp.gamma/100;
    %R.wt = R.wt/10^10;
    %algp.bathWavPenalty = algp.bathWavPenalty/100;
    %algp.bathPenalty = algp.bathPenalty/100;
    %keyboard
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Masked Component
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(algp.masknz) && algp.domasked
    
    % run theta_update for nonzero pixels, with no sparsity regularization.
    % we do this because we know that sparsity regularization will 
    % attenuate the map somewhat, so we need to relax that effect.
    costOld = Inf;
    cost = cost_eval(acqp.data,Gbrain,fBrain,Ac,theta,thetainit,R,algp.lam,algp.gamma,algp,Gbath,fBath,bathinit);
    itr = 0;
    fprintf('Masked iteration %d, cost = %f\n',itr,cost);
    while costOld-cost >= algp.stopThresh*costOld
                
        % update baseline in water bath
        dataBath = acqp.data(:) - Gbrain*(fBrain.*exp(1i*(Ac+theta))); % remove brain signal
        %fBath = f_update_bath(fBath,dataBath,acqp.L(algp.maskbath,:),acqp.kmask,Gall,algp);
        fBath = f_update_bath(dataBath,fBath,Gbath,bathinit,algp);
        
        % update baseline in brain
        dataBrain = acqp.data(:) - Gbath*fBath; % remove bath signal
        fBrain = f_update_brain(dataBrain,Ac,theta,acqp.L(algp.maskbrain,:),Gbrain);
                        
        % update poly coeffs
        c = c_update(dataBrain,A,c,theta,fBrain,Gbrain,algp);
        Ac = A*c; 
        
        % update temp shift
        theta = theta_update(dataBrain,Ac,theta,thetainit,fBrain,Gbrain,algp,algp.lam,algp.gamma,R,algp.masknz(algp.maskbrain),algp.sumMask);
        
        if algp.dofigs;figure(201);
            subplot(231); imagesc(embed(-real(theta),algp.maskbrain).'); axis image; title 'Estimated phase';colorbar
            subplot(232); imagesc(embed(imag(theta),algp.maskbrain).'); axis image; title 'Estimated Nepers';colorbar
            subplot(233); imagesc(embed(-real(theta) >= algp.maskthresh,algp.maskbrain).'); axis image; title 'Significant phase'
            subplot(234); im(embed(fBath,algp.maskbath));
            drawnow;
        end
        
        % calculate cost with updated parameters
        costOld = cost;
        cost = cost_eval(acqp.data,Gbrain,fBrain,Ac,theta,thetainit,R,algp.lam,algp.gamma,algp,Gbath,fBath,bathinit);
        
        itr = itr + 1;
        fprintf('Masked Iteration %d, cost = %f\n',itr,cost);
        
    end
end

% embed final results into full image matrix
theta = embed(theta,algp.maskbrain);
Ac = embed(Ac,algp.maskbrain);
f = embed(fBrain,algp.maskbrain) + embed(fBath,algp.maskbath);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Supporting Subfunctions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 
% Build the polynomial matrix
%
function A = buildA(mask,order)

if length(size(mask)) == 2 % build a 2D polynomial matrix
    
    [yc,xc] = meshgrid(linspace(-1/2,1/2,size(mask,2)), ...
        linspace(-1/2,1/2,size(mask,1)));
    yc = yc(:);
    xc = xc(:);
    A = [];
    for yp = 0:order
        for xp = 0:(order-yp)
            A = [A (xc.^xp).*(yc.^yp)];
        end
    end
    A = A(mask(:),:);
    
else % build a 3D polynomial matrix
    
    [zc,yc,xc] = meshgrid(linspace(-1/2,1/2,size(mask,3)), ...
        linspace(-1/2,1/2,size(mask,2)), linspace(-1/2,1/2,size(mask,1)));
    zc = zc(:);
    yc = yc(:);
    xc = xc(:);
    A = [];
    for yp = 0:order
        for xp = 0:(order-yp)
            for zp = 0:(order-(yp+xp))
                A = [A (xc.^xp).*(yc.^yp).^(zc.^zp)];
            end
        end
    end
    A = A(mask(:),:);
    
end

%
% Build the system matrices
%
function [Gbrain,Gbath,Gall] = buildG(Nc,k,fov,maskBrain,maskBath,mask)

if ~islogical(k) % non-cartesian
    
    % build system matrix
    if size(k,3) == 1 % 1 shot
        G = Gmri(k,mask,'fov',fov,'basis',{'dirac'});
    else % multishot
        nshot = size(k,3);
        for ii = 1:nshot % build a system matrix for each shot
            Gbrainsub{ii} = Gmri(k(:,:,ii),maskBrain,'fov',fov,'basis',{'dirac'});
            Gbathsub{ii} = Gmri(k(:,:,ii),maskBath,'fov',fov,'basis',{'dirac'});
            Gsub{ii} = Gmri(k(:,:,ii),mask,'fov',fov,'basis',{'dirac'});
        end
        Gbrain = block_fatrix(Gbrainsub,'type','col');
        Gbath = block_fatrix(Gbathsub,'type','col');
        Gall = block_fatrix(Gsub,'type','col');
    end
    %Gall = [];
        
else % cartesian
    
    Gbrain = Gmri_cart(k,maskBrain);
    Gbath = Gmri_cart(k,maskBath);
    Gall = Gmri_cart(true(size(k)));
        
end

if Nc > 1 % multiple coils; replicate the nufft's into a block-diag matrix
    for ii = 1:Nc
        tmp{ii} = G;
    end
    G = block_fatrix(tmp,'type','diag');
end


% 
% Evaluate cost
%
function cost = cost_eval(data,Gbrain,fBrain,Ac,theta,thetainit,R,lam,gamma,algp,Gbath,fBath,bathinit)

% get total estimated image
err = data(:) - Gbrain*(fBrain.*exp(1i*(Ac+theta)));
if exist('Gbath','var') % bath signal may already be subtracted from data in theta/c update funcs
    err = err - Gbath*fBath;
end
cost = 1/2*real(err'*err);

if exist('R','var')
  if ~isempty(R)
    cost = cost + R.penal(R,real(theta(:))) + R.penal(R,imag(theta(:)));
  end
end
if exist('gamma','var')
    cost = cost + 1/2*gamma*real((theta(:)-thetainit(:))'*(theta(:)-thetainit(:)));
end
if exist('lam','var')
    if ~algp.jointl1
        cost = cost - lam(1)*sum(real(theta(:))) + lam(2)*sum(imag(theta(:)));
    else
        cost = cost + lam*sum(sqrt(abs(theta(:)).^2 + algp.thetaEps));
    end
end

if exist('Gbath','var')
    bathPenaltyTemporal = 1/2*algp.bathPenalty*real((fBath(:)-bathinit(:))'*(fBath(:)-bathinit(:)));
    cost = cost + bathPenaltyTemporal;
    printf('Cost Elements:')
    printf('\t Error: %f',1/2*real(err'*err));
    printf('\t theta Roughness: %f',R.penal(R,real(theta(:))) + R.penal(R,imag(theta(:))));
    printf('\t theta temporal: %f',1/2*gamma*real((theta(:)-thetainit(:))'*(theta(:)-thetainit(:))));
    if ~algp.jointl1
        printf('\t theta sparsity: %f',- lam(1)*sum(real(theta(:))) + lam(2)*sum(imag(theta(:))));
    else
        printf('\t theta sparsity: %f',lam*sum(sqrt(abs(theta(:)).^2 + algp.thetaEps)));
    end
    printf('\t bath Temporal: %f',bathPenaltyTemporal);
end


%
% Update heat phase shift vector theta
%
function theta = theta_update(data,Ac,theta,thetainit,f,G,algp,lam,gammareg,R,masknz,sumMask)

% Polak-Ribiere PCG algorithm from JA Fessler's book, chapter 2, 11.7.13
g = [];
thresh = pi/1000;
for nn = 1:algp.thiters
    gold = g;
    g = gradcalc_theta(data,Ac,theta,thetainit,f,G,R,lam,gammareg,algp);
    if exist('masknz','var')
        g = g.*masknz;
    end
    if exist('sumMask','var') && exist('masknz','var')
        if sumMask == true
            g = sum(g(:))*masknz;
        end
    end
    if nn == 1
        dir = -g;
    else
        gamma = max(0,real(g'*(g-gold))/real(gold'*gold));
        dir = -g + gamma*dir;
    end
    %dir(dir > 0 & -theta < thresh) = 0; % Fessler 11.11.1
    dir(real(dir) > 0 & -real(theta(:,end)) < thresh) = 1i*imag(dir(real(dir) > 0 & -real(theta(:,end)) < thresh));
    dir(imag(dir) < 0 & imag(theta(:,end)) < thresh) = real(dir(imag(dir) < 0 & imag(theta(:,end)) < thresh));
    [t,breakOut] = stepcalc_theta(dir,data,Ac,theta,thetainit,f,G,R,lam,gammareg,100*min(1,pi/2/max(abs(dir))),g,algp);
    z = theta + t*dir;
    if any(real(z) > 0) || any(imag(z) < 0)
        %dir = z.*(z < 0) - theta;
        dir = real(z).*(real(z) < 0) - real(theta(:,end)) + 1i*(imag(z).*(imag(z) > 0) - imag(theta(:,end)));
        [t,breakOut] = stepcalc_theta(dir,data,Ac,theta,thetainit,f,G,R,lam,gammareg,1,g,algp);
    end
    if breakOut == true;break;end
    theta = theta + t*dir;
    %if t < eps;break;end
end

%
% Calculate gradient of cost wrt theta
%
function g = gradcalc_theta(data,Ac,theta,thetainit,f,G,R,lam,gamma,algp)

% data fidelity derivatives
img = f.*exp(1i*(Ac+theta));
%g = real(sum(reshape(1i*conj(img).*(G'*(data(:) - G*img)),[length(theta) Nc]),2));
g = 1i*conj(img).*(G'*(data(:) - G*img));
if ~algp.jointl1
    g = real(g) - lam(1) + 1i*(imag(g) + lam(2)); % l1 penalty derivatives 
else
    g = g + lam*theta./sqrt(abs(theta).^2 + algp.thetaEps);
end
g = g + gamma*(theta-thetainit);
if ~isempty(R) % roughness penalty derivatives
    g = g + R.cgrad(R,real(theta)) + 1i*R.cgrad(R,imag(theta));
end

%
% Calculate theta step size
%
function [t,breakOut] = stepcalc_theta(dir,data,Ac,theta,thetainit,f,G,R,lam,gamma,tmax,thetagrad,algp)

% use boyd's backtracking line search, which usually requires fewer cost evaluations

% calculate current cost
cost = cost_eval(data,G,f,Ac,theta,thetainit,R,lam,gamma,algp);

% line search to get step
costt = cost;
a = 0.5; b = 0.5; t = tmax/b;
while (costt > cost + a*t*real(thetagrad'*dir)) && t > 10^-6
    
    % reduce t
    t = b*t;
    
    % get test point
    thetat = theta + t*dir;
    
    % calculate cost of test point
    costt = cost_eval(data,G,f,Ac,thetat,thetainit,R,lam,gamma,algp);
    
end

if t == tmax/b % loop was never entered; return zero step
    t = 0;
end

if cost - costt >= 0.001*cost
    breakOut = false;
else
    breakOut = true;
end

% 
% Update polynomial coefficient vector c
% 
function c = c_update(data,A,c,theta,f,G,algp)

g = []; % gradient
for nn = 1:algp.citers
    gold = g;
    g = gradcalc_c(data,A,c,theta,f,G);
    if nn == 1
        dir = -g;
    else
        gamma = max(0,real(g'*(g-gold))/real(gold'*gold));
        dir = -g + gamma*dir;
    end
    alpha = stepcalc_c(dir,data,A,c,theta,f,G,min(1,pi/2/max(abs(dir(:)))),g);
    c = c + alpha*dir;
end


%
% Calculate gradient of cost wrt c
%
function g = gradcalc_c(data,A,c,theta,f,G)

Ac = A*c;
img = f.*exp(1i*(Ac+theta));
g = A'*real(1i*conj(img).*(G'*(data(:) - G*img)));


%
% Calculate step size for c
%
function t = stepcalc_c(dir,data,A,c,theta,f,G,tmax,cgrad,mask)

% use boyd's backtracking line search, which usually requires fewer cost evaluations

% calculate current cost
cost = cost_eval(data,G,f,A*c,theta);

% line search to get step
costt = cost;
a = 0.5; b = 0.5; t = tmax/b;
while (costt > cost + a*t*real(cgrad'*dir)) && t > 10^-6
    
    % reduce t
    t = b*t;
    
    % get test point
    ct = c + t*dir;
    
    % calculate cost of test point
    costt = cost_eval(data,G,f,A*ct,theta);
    
end

if t == tmax/b % loop was never entered; return zero step
    t = 0;
end



%
% Update baseline image estimates
%
function f = f_update_brain(data,Ac,theta,L,G)

Nc = size(data,2);

if size(L,2) > 1 % if more than one library image    
    
    % project library images to k-space
    for ii = 1:size(L,2)
        Lk(:,ii) = G*(L(:,ii).*repmat(exp(1i*(Ac+theta)),[Nc 1]));
    end
    LtL = real(Lk'*Lk);
    
    % set up constraints
    Ceq = ones(1,size(L,2));
    beq = 1;
    
    % set up cost
    c = -real(data(:)'*Lk);
    
    % solve
    options = optimset;options.MaxIter = 100000;options.Algorithm = 'active-set';
    wts = quadprog(double(LtL),double(c),[],[],Ceq,beq,zeros(size(L,2),1),ones(size(L,2),1),[],options);
    
    % get f
    f = L * wts;
    
else
    
    % only one baseline, so weight vector = 1;
    f = L;
    
end




function t = stepcalc_f(data,f,G,bathinit,algp,fgrad,dir,tmax)

% use boyd's backtracking line search, which usually requires fewer cost evaluations

% calculate current cost
err = data(:)-G*f(:);
%cost = 1/2*real(err'*err) + 1/2*algp.bathPenalty*real(Wf(:)'*Wf(:));
    %1/2*algp.bathPenalty*real(f(:)'*f(:));
cost = 1/2*real(err'*err) + ...
    1/2*algp.bathPenalty*real((f(:) - bathinit(:))'*(f(:) - bathinit(:)));
Gfbase = G*f(:);
Gfstep = G*dir(:);
%1/2*real(err'*err)
%algp.bathWavPenalty*sum(sqrt(abs(Wf(:)).^2+algp.bathEps))

% line search to get step
costt = cost;
a = 0.5; b = 0.5; t = tmax/b;
while (costt > cost + a*t*real(fgrad'*dir)) && t > 10^-6
    
    % reduce t
    t = b*t;
    
    % get test point
    ft = f + t*dir;
    
    % calculate cost of test point
    err = data(:)- Gfbase - t*Gfstep;%G*ft(:);
    costt = 1/2*real(err'*err) + ...
        1/2*algp.bathPenalty*real((ft(:) - bathinit(:))'*(ft(:) - bathinit(:))); 
    
end

if t == tmax/b % loop was never entered; return zero step
    t = 0;
end
    

%
% Update water bath
%
function f = f_update_bath(data,f,G,bathinit,algp)

[xS,~] = qpwls_pcg(f(:),G,1,data(:),0,0,1,algp.fBathIters,algp.maskbath);
f = xS(:,end);%embed(xS(:,end),algp.maskbath);

% % Polak-Ribiere PCG algorithm from JA Fessler's book, chapter 2, 11.7.13
% g = [];
% for nn = 1:algp.fBathIters
%     gold = g;
%     g = -(G'*(data(:)-G*f(:))) + algp.bathPenalty*(f(:) - bathinit(:));
%     if nn == 1
%         dir = -g;
%     else
%         gamma = max(0,real(g'*(g-gold))/real(gold'*gold));
%         dir = -g + gamma*dir;
%     end
%     t = stepcalc_f(data,f,G,bathinit,algp,g,dir,10000);
%     f = f + t*dir;
%     if t < eps;break;end % stop if step is insignificant
% end
