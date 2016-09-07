run ~/startup.m

load('testdata.mat','img','sampind','peakind','fov','mask','b','libind','dim');

img = conj(img); % complex conjugate all data so we get positive heat phase shifts

% temporary! shift so image is (mostly) in circular fov 
img = circshift(img,[0,5,0]);

libind = 2; % brain has hit steady state by second dynamic
mediannorm = median(abs(col(img(:,:,libind))));
img = img/mediannorm;

% synthesize a golden angle radial k-space trajectory
theta = 111.25;
fov = 28;
deltax = fov/dim;       % space between image samples [cm]
kmax = 1/(2*deltax);	% maximum kspace radius [1/cm]
projphi = 0;
ndim = 2;
nshot = ceil(pi*kmax*fov);
projline = [-sqrt(2)*dim/fov/2:1/fov:sqrt(2)*dim/fov/2]; 
k = zeros(length(projline),ndim,nshot); 
dimk = length(projline);

[xcmask,ycmask] = meshgrid(-dim/2:dim/2-1);
circmask = xcmask.^2 + ycmask.^2 < (dim/2)^2;

% kspace data samples for each projection line
d = []; 
for nline = 1:nshot   
  k(:,:,nline) = [projline.*cosd(projphi);-projline.*sind(projphi)]'; % kspace sample locations 
  projphi = projphi + theta; % update projection angle
  G = Gmri(k(:,:,nline),circmask,'fov',fov,'basis',{'dirac'}); % NUFFT object
  for jj = 1:size(img,3)     
    d(:,nline,jj) = G*col(img(:,:,jj)); % kspace data samples    
  end
end

% reconstruct baseline image
jj = libind;
niters = 25;
Grec = Gmri(reshape(permute(k(:,:,1:nshot),[2 1 3]),[2 dimk*nshot]).',circmask,'fov',fov,'basis',{'dirac'});
[xS,info] = qpwls_pcg(0*circmask,Grec,1,col(d(:,1:nshot,jj)),0,0,1,niters,circmask);
imglib = embed(xS(:,end),circmask);
L = imglib(:);

% define brain and bath masks
brainMask = circshift(b,[0 5]);
bathMask = abs(imglib.*~brainMask) > 1000/mediannorm;

% acceleration factor
accfactor = 4;
nshotacc = floor(nshot/accfactor);

thetakcs = [];%zeros([dim dim size(img,3)]);
thetakonly = [];
thetaBaseSub = [];
f = thetakcs; 
fk = thetakonly; 
thetakcs_max = [];%zeros(size(img,3),1);
thetaBaseSub_max = [];
thetakcs_mean = [];
thetaBaseSub_mean = [];
thetakonly_max = [];
thetakonly_mean = [];
hsmask = false(128);
hsmask(62:65,60:63) = true;

% stack data samples and k-space locations
dstack = d(:,:);
kstack = reshape(repmat(k,[1 1 1 size(img,3)]),[dimk ndim nshot*size(img,3)]);

% loop over dynamics
%for jj = nshot*(peakind-1)+1+nshot/2 %nshot*(libind+1)+1+nshot/2:nshot:size(dstack,2)-nshot/2;
for jj = nshot*(libind)+1+nshot/2:nshot:size(dstack,2)-nshot/2;

    indsfull = jj-nshot/2:jj+nshot/2-1;
    indsacc = jj+[-floor(nshotacc/2):floor(nshotacc/2)];
    
    % reconstruct fully-sampled dynamic image
    % [xS,info] = qpwls_pcg(0*circmask,Grec,1,col(d(:,1:nshot,jj)),0,0,1,niters,circmask);
    % Grec = Gmri(reshape(permute(kstack(:,:,jj:jj+nshot-1),[2 1 3]),[2 dimk*nshot]).',circmask,'fov',fov,'basis',{'dirac'});
    % [xS,info] = qpwls_pcg(0*circmask,Grec,1,col(dstack(:,jj:jj+nshot-1)),0,0,1,niters,circmask);
    Grec = Gmri(reshape(permute(kstack(:,:,indsfull),[2 1 3]),[2 dimk*nshot]).',circmask,'fov',fov,'basis',{'dirac'});
    [xS,info] = qpwls_pcg(0*circmask,Grec,1,col(dstack(:,indsfull)),0,0,1,niters,circmask);
    imghot = embed(xS(:,end),circmask);
    
    dtmp = dstack(:,indsacc);
    ktmp = kstack(:,:,indsacc);

    % run k-space hybrid code
    
    % acquisition parameters
    acqp.data = dtmp(:);        % k-space data samples
    acqp.fov = fov;             % field of view
    acqp.k = ktmp;              % k-space sampling mask % WAG REDFLAG: This is not an actual mask, but instead is the k-space sample locs, which don't seem to be the same as the mask
    acqp.L = L(:);        	    % baseline 'library'
    acqp.mask = circmask;       % mask
    acqp.kmask = ktmp;          % mask of sampled k-space locations
    
    % algorithm parameters
    algp.dofigs = 0;            % show figures
    algp.order = 1;             % polynomial order
    algp.lam = 10^-6;           % sparsity regularization parameter
    algp.beta = 10^-6;          % roughness regularization parameter
    algp.gamma = 10^-5;         % temporal regularization parameter
    algp.modeltest = 0;         % model test
    algp.maskthresh = 0.01;     % phase shift threshold
    algp.domasked = 1;          % whether to run masked update
    algp.maskbath = bathMask;   % waterbath mask
    algp.maskbrain = logical(brainMask); % mask of brain
    algp.stopThresh = 10^-3;    % stop threshold (= fraction of previous cost that cost difference must be > than each iter)
    algp.bathPenalty = 0;
    algp.fBathIters = 5;
    algp.bathEps = 10^-10;
    algp.thetaEps = 10^-10;
    algp.bathWavPenalty = 10^-7;
    algp.sumMask = true; % do a DC relaxation in the masked iterations
    algp.jointl1 = true; % jointly sparsity-penalize the real and imaginary parts of theta
    
    % cs masked k-space hybrid recon
    % bathinit = bathMask.*img(:,:,jj-1)./mediannorm;
    if jj == nshot*(libind+1)+1+nshot/2%libind + 1
        thetainit = zeros(dim);
        bathinit = zeros(dim);
        algp.bathPenalty = 0;
        algp.gamma = 0;
    else
        bathinit = bathMask.*f(:,:,end);% bathMask.*f(:,:,jj-1);
        thetainit = thetakcs(:,:,end);
    end
    [thetakcs(:,:,end+1),~,~,f(:,:,end+1),Ac,~] = kspace_hybrid_thermo_mask_svcs(acqp,thetainit,bathinit,algp);
    %[thetakcs(:,:,end+1),~,~,f(:,:,end+1),Ac,~] = kspace_hybrid_thermo_mask_v3(acqp,thetainit,bathinit,algp);
    thetaBaseSub(:,:,end+1) = angle(imglib.*conj(imghot).*exp(1i*Ac));
    
    % plot peak and mean phase in hot spot region
    tmpkcs = real(thetakcs(:,:,end));
    tmpBaseSub = thetaBaseSub(:,:,end);
    thetakcs_max(end+1) = max(-tmpkcs(hsmask));
    thetaBaseSub_max(end+1) = max(tmpBaseSub(hsmask));
    thetakcs_mean(end+1) = mean(-tmpkcs(hsmask));
    thetaBaseSub_mean(end+1) = mean(tmpBaseSub(hsmask));

    % k-space only
    algp.maskbrain = logical(circmask); % mask of brain
    algp.maskbath = logical(0*circmask);
    if jj == nshot*(libind)+1+nshot/2%libind + 1
        thetainit = zeros(dim);
    else
        thetainit = thetakonly(:,:,end);
    end
    bathinit = zeros(dim);
    [thetakonly(:,:,end+1),~,~,fk(:,:,end+1),Ack,~] = kspace_hybrid_thermo_mask_svcs(acqp,thetainit,bathinit,algp);
    
    % plot peak and mean phase in hot spot region
    tmpkonly = real(thetakonly(:,:,end));
    thetakonly_max(end+1) = max(-tmpkonly(hsmask));
    thetakonly_mean(end+1) = mean(-tmpkonly(hsmask));
    
    % compare to baseline subtraction with same field drift correction
    figure; subplot(121), im([brainMask.'.*thetaBaseSub(:,:,end).' brainMask.'.*-real(thetakcs(:,:,end)).'].',[0 0.5]);
    subplot(122),plot([thetaBaseSub_max thetakcs_max thetaBaseSub_mean thetakcs_mean]),axis square;
    legend('Base sub, max','kseg, max','Base sub, mean','kseg, mean');
    drawnow
    
end


return
% save(sprintf('radialreconacc%d',accfactor))
% exit



%% plot results

load radialreconacc4_all


TE = 0.012772;
B0 = 3;
alpha = 0.01;
gamma = 2*pi*42.57;
ct = -1/TE/B0/alpha/gamma;


% plot maps 
dispinds = [5,8,10,15];
figure; 
xinds=30:99; yinds=[15:104]+5;
maskb = brainMask.*abs(img(:,:,peakind))>3500/mediannorm;
for inds = 1:length(dispinds)
    jj = dispinds(inds)-2;
    disp (jj)
    subplot(1,length(dispinds),inds)
    im([-ct*thetaBaseSub(xinds,yinds,jj).'.*maskb(xinds,yinds).';ct*real(thetakonly(xinds,yinds,jj)).'.*maskb(xinds,yinds).';ct*real(thetakcs(xinds,yinds,jj)).'.*maskb(xinds,yinds).'].',[0 5]);
    colormap jet; axis off
    title(sprintf('4x: dynamic %d',dispinds(inds)))
    %pause;
end

% plot error
errk = max(abs(-ct*repmat(maskb,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetakonly))),[],3);
errkcs = max(abs(-ct*repmat(maskb,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetakcs))),[],3);
figure; im([errk(xinds,yinds) errkcs(xinds,yinds)],[0 10]); colormap jet


% plot peak temp change in hot spot
figure; 
% subplot(211); 
plot(-ct*thetaBaseSub_max,'k'); hold on; plot(-ct*thetakonly_max,'g'); plot(-ct*thetakcs_max,'r');
title('Peak temperature in hot spot (16 voxels)'); 
legend('fully sampled baseline subtraction','kspace everywhere, 4x','kspace brain/NLCG bath, 4x2D');
xlabel('dynamic'); ylabel('Temperature (C)'); 

% plot mean temp change in hot spot
% figure; plot(-ct*thetaBaseSub_mean,'k','LineWidth',2); hold on; 
% plot(-ct*thetakonly_mean,':','LineWidth',2,'Color',[0.1 0.5 0.2]);
% plot(-ct*thetakcs_mean,'r-.','LineWidth',2); 

figure; plot([3:26],[0 -ct*thetaBaseSub_mean],'k','LineWidth',2); hold on; plot([3:26],[0 -ct*thetakonly_mean],':','LineWidth',2,'Color',[0.1 0.5 0.2]); plot([3:26],[0 -ct*thetakcs_mean],'r-.','LineWidth',2); 
xlim([3,26])
title('Mean temperature in hot spot (16 voxels)'); 
legend('fully sampled baseline subtraction','kspace everywhere, 4x','kspace brain/NLCG bath, 4x');
xlabel('dynamic'); ylabel('Temperature (C)'); 

% fix dynamic offset
tmpsub = zeros(1,26);
tmpk = zeros(1,26);
tmpkcs = zeros(1,26);
tmpsub(:,4:end) = thetaBaseSub_mean;
tmpk(:,4:end) = thetakonly_mean;
tmpkcs(:,4:end) = thetakcs_mean;
figure; plot(-ct*tmpsub,'k','LineWidth',2); hold on; 
plot(-ct*tmpk,':','LineWidth',2,'Color',[0.1 0.5 0.2]);
plot(-ct*tmpkcs,'r-.','LineWidth',2); 


% run libind+1 dynamic, because skipped it the first time
jj = nshot*(libind)+1+nshot/2;

indsfull = jj-nshot/2:jj+nshot/2-1;
indsacc = jj+[-floor(nshotacc/2):floor(nshotacc/2)];

dtmp = dstack(:,indsacc);
ktmp = kstack(:,:,indsacc);

    % run k-space hybrid code    
    % acquisition parameters
    acqp.data = dtmp(:);        % k-space data samples
    acqp.fov = fov;             % field of view
    acqp.k = ktmp;              % k-space sampling mask % WAG REDFLAG: This is not an actual mask, but instead is the k-space sample locs, which don't seem to be the same as the mask
    acqp.L = L(:);        	    % baseline 'library'
    acqp.mask = circmask;       % mask
    acqp.kmask = ktmp;%kmask;         % mask of sampled k-space locations
    % algorithm parameters
    algp.dofigs = 0;            % show figures
    algp.order = 1;             % polynomial order
    algp.lam = 10^-6;% 10^-3.5];     % sparsity regularization parameter
    algp.beta = 10^-6;%10^-5.5;%10^-5.75;%4.5;         % roughness regularization parameter
    algp.gamma = 10^-5;%10^-3.25;         % temporal regularization parameter
    algp.modeltest = 0;         % model test
    algp.maskthresh = 0.01;     % phase shift threshold
    algp.domasked = 1;          % whether to run masked update
    algp.maskbath = bathMask; % waterbath mask
    algp.maskbrain = logical(brainMask); % mask of brain
    algp.stopThresh = 10^-3;    % stop threshold (= fraction of previous cost that cost difference must be > than each iter)
    algp.bathPenalty = 0;%10^-8.25;
    algp.fBathIters = 5;
    algp.bathEps = 10^-10;
    algp.thetaEps = 10^-10;
    algp.bathWavPenalty = 10^-7;
    algp.sumMask = true; % do a DC relaxation in the masked iterations
    algp.jointl1 = true; % jointly sparsity-penalize the real and imaginary parts of theta
    
    % cs masked k-space hybrid recon
    thetainit = zeros(dim);
    bathinit = zeros(dim);
    algp.bathPenalty = 0;
    algp.gamma = 0;
    [thetakcs1,~,~,f1,Ac1,~] = kspace_hybrid_thermo_mask_svcs(acqp,thetainit,bathinit,algp);
    
    % k-space only
    algp.maskbrain = logical(circmask); % mask of brain
    algp.maskbath = logical(0*circmask);
    [thetakonly1,~,~,fk1,Ack1,~] = kspace_hybrid_thermo_mask_svcs(acqp,thetainit,bathinit,algp);

    Grec = Gmri(reshape(permute(kstack(:,:,indsfull),[2 1 3]),[2 dimk*nshot]).',circmask,'fov',fov,'basis',{'dirac'});
    [xS,info] = qpwls_pcg(0*circmask,Grec,1,col(dstack(:,indsfull)),0,0,1,niters,circmask);
    imghot = embed(xS(:,end),circmask);

    thetaBaseSub1 = angle(imglib.*conj(imghot).*exp(1i*Ac1));

    
    
    
    thetaBaseSub_new = zeros(dim,dim,size(img,3));
    thetakonly_new = zeros(dim,dim,size(img,3));
    thetakcs_new = zeros(dim,dim,size(img,3));
    f_new = zeros(dim,dim,size(img,3)); 
    fk_new = zeros(dim,dim,size(img,3)); 
    
    thetaBaseSub_new(:,:,libind+1) = thetaBaseSub1;
    thetakonly_new(:,:,libind+1) = thetakonly1; 
    thetakcs_new(:,:,libind+1) = thetakcs1; 
    
    thetaBaseSub_new(:,:,libind+2:end-1) = thetaBaseSub(:,:,2:end);
    thetakonly_new(:,:,libind+2:end-1) = thetakonly(:,:,2:end);
    thetakcs_new(:,:,libind+2:end-1) = thetakcs(:,:,2:end);
    
    f_new(:,:,libind+1) = f1;
    fk_new(:,:,libind+1) = fk1; 
    f_new(:,:,libind+2:end-1) = f(:,:,2:end);
    fk_new(:,:,libind+2:end-1) = fk(:,:,2:end);
    
    
    % plot peak and mean phase in hot spot region
    thetakcs_max_new = zeros(size(img,3),1);
    thetakcs_mean_new = zeros(size(img,3),1);
    thetakcs_max_new(libind+1) = max(-real(thetakcs1(hsmask)));
    thetakcs_mean_new(libind+1) = mean(-real(thetakcs1(hsmask)));
    thetakcs_max_new(libind+2:end-1) = thetakcs_max;
    thetakcs_mean_new(libind+2:end-1) = thetakcs_mean;
    
    thetaBaseSub_max_new = zeros(size(img,3),1);    
    thetaBaseSub_mean_new = zeros(size(img,3),1);
    thetaBaseSub_max_new(libind+1) = max(thetaBaseSub1(hsmask));
    thetaBaseSub_mean_new(libind+1) = mean(thetaBaseSub1(hsmask));
    thetaBaseSub_max_new(libind+2:end-1) = thetaBaseSub_max;
    thetaBaseSub_mean_new(libind+2:end-1) = thetaBaseSub_mean;
    
    tmpkonly = real(thetakonly1);
    thetakonly_max_new = zeros(size(img,3),1);
    thetakonly_mean_new = zeros(size(img,3),1);
    thetakonly_max_new(libind+1) = max(-tmpkonly(hsmask));
    thetakonly_mean_new(libind+1) = mean(-tmpkonly(hsmask));
    thetakonly_max_new(libind+2:end-1) = thetakonly_max;
    thetakonly_mean_new(libind+2:end-1) = thetakonly_mean;
    
        
    thetaBaseSub = thetaBaseSub_new;
    thetakonly = thetakonly_new;
    thetakcs = thetakcs_new;
    f = f_new;
    fk = fk_new;
    thetakonly_max = thetakonly_max_new;    
    thetakonly_mean = thetakonly_mean_new;
    thetakcs_max = thetakcs_max_new;    
    thetakcs_mean = thetakcs_mean_new;
    thetaBaseSub_max = thetaBaseSub_max_new;    
    thetaBaseSub_mean = thetaBaseSub_mean_new;
    clear *_new
    
    save radialreconacc4_all
    
    figure; plot(-ct*thetaBaseSub_mean,'k','LineWidth',2); hold on;
    plot(-ct*thetakonly_mean,':','LineWidth',2,'Color',[0.1 0.5 0.2]); 
    plot(-ct*thetakcs_mean,'r-.','LineWidth',2);
    
plot(dispinds(1),0,'ko','MarkerSize',5)
plot(dispinds(2),0,'ko','MarkerSize',5)
plot(dispinds(3),0,'ko','MarkerSize',5)
plot(dispinds(4),0,'ko','MarkerSize',5)
    