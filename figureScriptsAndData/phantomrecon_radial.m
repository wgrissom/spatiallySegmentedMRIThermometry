run ~/startup.m
% run conventional k-space and segmented k-space reconstructions using golden angle radial sampling

load('phantomdata.mat');

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
 
% kspace data samples for each projection line
d = []; 
for nline = 1:nshot   
  k(:,:,nline) = [projline.*cosd(projphi);-projline.*sind(projphi)]'; % kspace sample locations 
  projphi = projphi + theta; % update projection angle
  G = Gmri(k(:,:,nline),mask,'fov',fov,'basis',{'dirac'}); % NUFFT object
  for jj = 1:size(img,3)     
    d(:,nline,jj) = G*col(img(:,:,jj)); % kspace data samples    
  end
end

% reconstruct baseline image
jj = libind;
niters = 25;
Grec = Gmri(reshape(permute(k(:,:,1:nshot),[2 1 3]),[2 dimk*nshot]).',mask,'fov',fov,'basis',{'dirac'});
[xS,info] = qpwls_pcg(0*mask(:),Grec,1,col(d(:,1:nshot,jj)),0,0,1,niters,mask(:));
imglib = embed(xS(:,end),mask);
L = imglib(:);

% define brain and bath masks
brainMask = imgbath(:,:,1) == 0;
bathMask = abs(imgbath(:,:,1)) > 1000;

% acceleration factor
accfactor = 4;
nshotacc = floor(nshot/accfactor);

thetakcs = [];
thetak = [];
thetaBaseSub = [];
f = thetakcs; 
fk = thetak; 
thetakcs_max = [];
thetaBaseSub_max = [];
thetakcs_mean = [];
thetaBaseSub_mean = [];
thetak_max = [];
thetak_mean = [];
hsmask = false(128);
hsmask(60:67,53:60) = true;

% stack data samples and k-space locations
dstack = d(:,:);
kstack = reshape(repmat(k,[1 1 1 size(img,3)]),[dimk ndim nshot*size(img,3)]);

% loop over dynamics
%for jj = nshot*(peakind-1)+1+nshot/2 %
for jj = nshot*(libind)+1+nshot/2:nshot:size(dstack,2)+1-nshot/2;

    indsfull = jj-nshot/2:jj+nshot/2-1;
    indsacc = jj+[-floor(nshotacc/2):floor(nshotacc/2)-1];
    
    % reconstruct fully-sampled dynamic image
    % [xS,info] = qpwls_pcg(0*circmask,Grec,1,col(d(:,1:nshot,jj)),0,0,1,niters,circmask);
    % Grec = Gmri(reshape(permute(kstack(:,:,jj:jj+nshot-1),[2 1 3]),[2 dimk*nshot]).',circmask,'fov',fov,'basis',{'dirac'});
    % [xS,info] = qpwls_pcg(0*circmask,Grec,1,col(dstack(:,jj:jj+nshot-1)),0,0,1,niters,circmask);
    Grec = Gmri(reshape(permute(kstack(:,:,indsfull),[2 1 3]),[2 dimk*nshot]).',mask,'fov',fov,'basis',{'dirac'});
    [xS,info] = qpwls_pcg(0*mask(:),Grec,1,col(dstack(:,indsfull)),0,0,1,niters,mask(:));
    imghot = embed(xS(:,end),mask);
    
    dtmp = dstack(:,indsacc);
    ktmp = kstack(:,:,indsacc);

    % run k-space hybrid code
    
    % acquisition parameters
    acqp.data = dtmp(:);        % k-space data samples
    acqp.fov = fov;             % field of view
    acqp.k = ktmp; 		% mask of sampled k-space locations
    acqp.L = L(:);        	% baseline 'library'
    acqp.mask = mask;           % mask
    
    % algorithm parameters
    algp.dofigs = 0;            % show figures (0 -> no, 1 -> yes)
    algp.order = 1;             % polynomial order to estimate background phase drift
    algp.lam = 10^-10;          % sparsity regularization parameter
    algp.beta = 10^-4.4;        % roughness regularization parameter
    algp.gamma = 0;             % temporal regularization parameter
    algp.modeltest = 0;         % model test
    algp.maskthresh = 0.01;     % phase shift threshold
    algp.domasked = 1;          % whether to run masked update
    algp.maskbath = bathMask;   % mask of water bath region
    algp.maskbrain = brainMask; % mask of brain region
    algp.stopThresh = 10^-3;    % stop threshold (= fraction of previous cost that cost difference must be > than each iter)
    algp.stepThresh = 0.001;    % threshold to break out of line search in NLCG algorithm
    algp.bathPenalty = 0;       % bath roughness penalty
    algp.fBathIters = 4;        % # bath CG iters per outer iteration
    algp.thetaEps = 10^-10;     % theta l1 penalty offset
    algp.sumMask = true; 	% do a DC relaxation in the masked iterations
    algp.jointl1 = true; 	% jointly sparsity-penalize the real and imaginary parts of theta
    algp.updateBath = 1;        % update image in water bath (0 -> no, 1 -> yes)
    algp.bathRecon = 'CG';      % reconstruction method in water bath ('CG','NLCG')
    algp.bathinit = zeros(dim); % initial estimate for bath image; requires algp.updateBath
    algp.thetainit = zeros(dim);% initial estimate for heat phase shift

    % spatially-segmented k-space hybrid temperature reconstruction
    [thetakcs(:,:,end+1),~,~,f(:,:,end+1),Ac,~] = kspace_hybrid_thermo_sseg(acqp,algp);

    % whole-image k-space hybrid temperature reconstruction
    algp.updateBath = 0;
    [thetak(:,:,end+1),~,~,fk(:,:,end+1),Ack,~] = kspace_hybrid_thermo_sseg(acqp,algp);

    % fully-sampled baseline subtraction temperature reconstruction with phase drift correction
    thetaBaseSub(:,:,end+1) = angle(imglib.*conj(imghot).*exp(1i*Ac));
    
    % plot peak and mean phase in hot spot region
    tmpkcs = real(thetakcs(:,:,end));
    tmpBaseSub = thetaBaseSub(:,:,end);    
    tmpk = real(thetak(:,:,end));
    thetakcs_max(end+1) = max(-tmpkcs(hsmask));
    thetak_max(end+1) = max(-tmpk(hsmask));
    thetaBaseSub_max(end+1) = max(tmpBaseSub(hsmask));
    thetakcs_mean(end+1) = mean(-tmpkcs(hsmask));
    thetak_mean(end+1) = mean(-tmpk(hsmask));
    thetaBaseSub_mean(end+1) = mean(tmpBaseSub(hsmask));
    
end

save(sprintf('phantomresults_radialacc%d',accfactor))

return


%% plot results
fnames = {'garadialacc1_again.mat','garadialacc2_again.mat',...
    'garadialacc3_again.mat','garadialacc4_again.mat'};
thetaBaseSub = zeros(128,128,27,length(fnames));
thetakcs = zeros(128,128,27,length(fnames));
thetak = zeros(128,128,27,length(fnames));
for nn = 1:length(fnames); 
    S = load(fnames{nn});
    accfactor(nn) = S.accfactor;
    thetaBaseSub(:,:,2:27,nn) = S.thetaBaseSub;
    thetakcs(:,:,2:27,nn) = S.thetakcs;
    thetak(:,:,2:27,nn) = S.thetak;
end
thetaBase = thetaBaseSub(:,:,:,1); 
thetaBase(thetaBase < 0) = 0;
hsmask = S.hsmask;
maskb = S.brainMask.*abs(S.img(:,:,S.peakind))>3500/S.mediannorm;


for nn = 1:length(fnames)
    for jj = S.libind+1:size(S.img,3)
    % plot peak and mean phase in hot spot region
    tmpkcs = real(thetakcs(:,:,jj,nn));
    tmpk = real(thetak(:,:,jj,nn));
    tmpBaseSub = thetaBaseSub(:,:,jj,nn);
    thetakcs_max(nn,jj) = max(-tmpkcs(hsmask));
    thetak_max(nn,jj) = max(-tmpk(hsmask));
    thetaBaseSub_max(nn,jj) = max(tmpBaseSub(hsmask));
    thetakcs_mean(nn,jj) = mean(-tmpkcs(hsmask));
    thetak_mean(nn,jj) = mean(-tmpk(hsmask));
    thetaBaseSub_mean(nn,jj) = mean(tmpBaseSub(hsmask));
    end
end
    

TE = 0.012772;
B0 = 3;
alpha = 0.01;
gamma = 2*pi*42.57;
ct = -1/TE/B0/alpha/gamma;


% plot maps 
dispinds = [5,8,10,15];
figure; 
xinds=30:99; yinds=[15:104]+5;
for nn = 1:length(fnames)
  for inds = 1:length(dispinds)
    figure(nn)
    jj = dispinds(inds);
    disp (jj)
    subplot(1,length(dispinds),inds)
    im([-ct*thetaBaseSub(xinds,yinds,jj,nn).'.*maskb(xinds,yinds).';ct*real(thetak(xinds,yinds,jj,nn)).'.*maskb(xinds,yinds).';ct*real(thetakcs(xinds,yinds,jj,nn)).'.*maskb(xinds,yinds).'].',[0 5]);
    colormap jet; axis off
    title(sprintf('%gx ga radial: dynamic %d',accfactor(nn), dispinds(inds)))
    %pause;
  end
end

for nn = 1:length(fnames);
    thetakrmse(nn) = rmse(-ct*thetaBase,ct*real(thetak(:,:,:,nn)),repmat(maskb,[1 1 size(S.img,3)]))/sqrt(sum(maskb(:))*size(S.img,3));
    thetakcsrmse(nn) = rmse(-ct*thetaBase,ct*real(thetakcs(:,:,:,nn)),repmat(maskb,[1 1 size(S.img,3)]))/sqrt(sum(maskb(:))*size(S.img,3));
    thetakrmse_hs(nn) = rmse(-ct*thetaBase,ct*real(thetak(:,:,:,nn)),repmat(hsmask,[1 1 size(S.img,3)]))/sqrt(sum(hsmask(:))*size(S.img,3));
    thetakcsrmse_hs(nn) = rmse(-ct*thetaBase,ct*real(thetakcs(:,:,:,nn)),repmat(hsmask,[1 1 size(S.img,3)]))/sqrt(sum(hsmask(:))*size(S.img,3));
end

 
figure(43);
subplot(121); bar(thetakrmse,0.4); grid on
set(gca,'XTickLabel',accfactor)
hold on; bar(thetakrmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2)
ylabel(sprintf('RMSE (%cC)',char(176)))
xlabel('acceleration factor')
title('RMSE, k-space everywhere')
subplot(122); bar(thetakcsrmse,0.4); grid on
set(gca,'XTickLabel',accfactor)
hold on; bar(thetakcsrmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2)
ylabel(sprintf('RMSE (%cC)',char(176)))
xlabel('acceleration factor')
title('RMSE, k-space brain / CG bath')
ylim([0 2])
legend('brain','hot spot')

% % plot max error
errk = max(abs(-ct*repmat(maskb,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetak))),[],3);
errkcs = max(abs(-ct*repmat(maskb,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetakcs))),[],3);
%figure; im([errk(xinds,yinds) errkcs(xinds,yinds)],[0 10]); colormap jet

% plot fully-sampled, k-space everywhere (acc1), and all k-space brain/CG bath
figure(800);
subplot(211);
plot(-ct*thetaBaseSub_mean(1,:),'k','LineWidth',2); 
hold on; plot(-ct*thetak_mean(1,:),'k--','LineWidth',2); 
plot(-ct*thetakcs_mean(1,:),':^','LineWidth',2,'Color',[0 0.7 0.9]);
plot(-ct*thetakcs_mean(2,:),':+','LineWidth',2,'Color',[1 0.749 0]);
plot(-ct*thetakcs_mean(3,:),':s','LineWidth',2,'Color',[0.75 0.25 0]);%[0.91 0.41 0.17]);
plot(-ct*thetakcs_mean(4,:),':x','LineWidth',2,'Color',[0.9 0.4 0.9]);
%plot(-ct*thetakcs_mean(4,:),':*','LineWidth',2,'MarkerSize',2,'Color',[0 0.4 0.2]);%[0.1 0.5 0.2]);
title(sprintf('Mean temperature in hot spot'));
% legend('fully sampled baseline subtraction',sprintf('kspace everywhere, %gx',accfactor(1)),...
%     sprintf('spatially-segmented, %gx',accfactor(1)),sprintf('spatially-segmented, %gx',accfactor(2)),...
%     sprintf('spatially-segmented, %gx',accfactor(3)),sprintf('spatially-segmented, %gx',accfactor(4)));
xlabel('dynamic'); ylabel('Temperature (C)');
grid on;
xlim([0 size(thetak,3)])
plot(dispinds(1),0,'ko','MarkerSize',5)
plot(dispinds(2),0,'ko','MarkerSize',5)
plot(dispinds(3),0,'ko','MarkerSize',5)
plot(dispinds(4),0,'ko','MarkerSize',5)
subplot(212);
plot(-ct*thetaBaseSub_max(1,:),'k','LineWidth',2); 
hold on; plot(-ct*thetak_max(1,:),'k--','LineWidth',2); 
plot(-ct*thetakcs_max(1,:),':^','LineWidth',2,'Color',[0 0.7 0.9]);
plot(-ct*thetakcs_max(2,:),':+','LineWidth',2,'Color',[1 0.749 0]);%[0 0.447 0.741]);
plot(-ct*thetakcs_max(3,:),':s','LineWidth',2,'Color',[0.75 0.25 0]);
plot(-ct*thetakcs_max(4,:),':x','LineWidth',2,'Color',[0.9 0.4 0.9]);
%plot(-ct*thetakcs_max(5,:),':*','LineWidth',2,'Color',[0 0.4 0.2],'MarkerSize',2);%[0.15 0.45 0.15]);
title(sprintf('Max temperature in hot spot'));
legend('fully sampled',sprintf('kspace everywhere, %gx',accfactor(1)),...
    sprintf('spatially-segmented, %gx',accfactor(1)),sprintf('spatially-segmented, %gx',accfactor(2)),...
    sprintf('spatially-segmented, %gx',accfactor(3)),sprintf('spatially-segmented, %gx',accfactor(4)));
xlabel('dynamic'); ylabel('Temperature (C)');
grid on;
xlim([0 size(thetak,3)])
plot(dispinds(1),0,'ko','MarkerSize',5)
plot(dispinds(2),0,'ko','MarkerSize',5)
plot(dispinds(3),0,'ko','MarkerSize',5)
plot(dispinds(4),0,'ko','MarkerSize',5)

