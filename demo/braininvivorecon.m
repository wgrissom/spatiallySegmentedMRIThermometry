run ~/startup.m
% select the undersampling pattern to use
sampPattern = '1x,1D';%'2x,1D'; % '3x,1D', '4x,1D'

% human 8-coil data
load braininvivodata.mat % image series, brain ROI mask, and k-space sampling masks

dim = size(img,1);
Nc = size(img,3);
Ndyn = size(img,4);
te = 0.01270;
fov = 28;

bathMask = ~brainMask;
mask = true(dim);
% hot spot mask for calculating mean and max heat shift phase
hsmask = false(128);
hsmask(63:66,63:66) = true;

libInd = 2; % brain appears to hit steady state by first dynamic
NcgIters = 20; % CG image recon iterations

switch sampPattern
  case '1x,1D'
    % use fully sampled pattern
    kmask = logical(ones(dim));
  case '2x,1D'
    % use 1D 2x-undersampled pattern
    kmask = kmask2x1d;
  case '3x,1D'
    % use 1D 3x-undersampled pattern
    kmask = kmask3x1d;
  case '4x,1D'
    % use 1D 4x-undersampled pattern
    kmask = kmask4x1d;
end

% calculate raw phase differences (no drift corr), for reference
phsDiff = -squeeze(sum(angle(img.*conj(repmat(img(:,:,:,libInd),[1 1 1 Ndyn]))).*abs(img).^2,3)./sum(abs(img).^2,3));

% get the baseline image
imgLib = img(:,:,:,libInd); 
mediannorm = median(abs(imgLib(:)));
imgLib = imgLib/mediannorm;
L = imgLib; % for the k-space hybrid recon

% get sensitivities from average image across dynamics
imgTmp = squeeze(mean(img,4));
sens = imgTmp./repmat(ssq(imgTmp,3),[1 1 Nc]);
sens = sens .* (abs(imgTmp) > 0.01*max(abs(imgTmp(:))));

% allocate storage for results
thetakcs = zeros([dim dim Ndyn]);
thetaBaseSub = thetakcs;
thetaSENSE = thetakcs;
f = zeros([dim dim Nc Ndyn]); 
thetakcs_max = zeros(Ndyn,1);
thetaBaseSub_max = zeros(Ndyn,1);
thetaSENSE_max = zeros(Ndyn,1);
thetakcs_mean = zeros(Ndyn,1);
thetaBaseSub_mean = zeros(Ndyn,1);
thetaSENSE_mean = zeros(Ndyn,1);


for jj = libInd+1:Ndyn % loop over dynamics

    % dynamic image
    imgHot = img(:,:,:,jj)/mediannorm;

    % Build system matrix for SENSE recons using this dynamic's k-space
    % sampling mask
    G = Gmri_cart(kmask,mask); % undersampled FFT operator - same as will be used in recon
    % build a total G operator for CG-SENSE recon
    GS = {};
    for ii = 1:Nc
        S = diag_sp(col(sens(:,:,ii)));
        GS{ii} = block_fatrix({G,S},'type','mult');
    end
    GS = block_fatrix(GS,'type','col');

    % get the data back from the coil images
    data = [];
    for ii = 1:Nc
        data(:,ii) = G*col(imgHot(:,:,ii));
    end

    % do a CG-SENSE recon
    [xS,info] = qpwls_pcg(zeros(dim*dim,1),GS,1,data(:),0,0,1,NcgIters,true(dim));
    imgSENSE(:,:,:,jj) = repmat(reshape(xS(:,end),[dim dim]),[1 1 Nc]).*sens;

    % run k-space hybrid

    % acquisition parameters
    acqp.data = data; 	        	% k-space data samples
    acqp.fov = fov;             	% field of view
    acqp.k = kmask;             	% mask of sampled k-space locations
    acqp.L = L(:);        		% baseline 'library'
    acqp.mask = mask;           	% mask

    % algorithm parameters
    algp.dofigs = 0;            	% show figures (0 -> no, 1 -> yes)
    algp.order = 3;             	% polynomial order to estimate background phase drift
    algp.lam = 10^-6.265;       	% sparsity regularization parameter
    algp.beta = 10^-20;         	% roughness regularization parameter
    algp.gamma = 0;             	% temporal regularization parameter
    algp.modeltest = 0;         	% model test
    algp.maskthresh = 0.01;     	% phase shift threshold
    algp.domasked = 1;          	% whether to run masked update
    algp.maskbath = bathMask;   	% mask of water bath region
    algp.maskbrain = brainMask; 	% mask of brain region
    algp.stopThresh = 10^-3;    	% stop threshold (= fraction of previous cost that cost difference must be > than each iter)
    algp.stepThresh = 0.00001;  	% threshold to break out of line search in NLCG algorithm
    algp.sens = sens;           	% coil sensitivity maps
    algp.bathPenalty = 0;       	% bath roughness penalty
    algp.fBathIters = 2;        	% # bath CG iters per outer iteration
    algp.thetaEps = 10^-12;     	% theta l1 penalty offset
    algp.sumMask = true;        	% do a DC relaxation in the masked iterations
    algp.jointl1 = true;        	% jointly sparsity-penalize the real and imaginary parts of theta
    algp.updateBath = 1;        	% update image in water bath (0 -> no, 1 -> yes)
    algp.bathRecon = 'CG';      	% reconstruction method in water bath ('CG','NLCG')
    algp.bathinit = zeros(dim,dim,Nc); 	% initial estimate for bath image; requires algp.updateBath
    algp.thetainit = zeros(dim);	% initial estimate for heat phase shift

    % spatially-segmented k-space hybrid temperature reconstruction
    [thetakcs(:,:,jj),~,~,f(:,:,:,jj),Ac,~] = kspace_hybrid_thermo_sseg(acqp,algp);

    % whole-image k-space hybrid temperature reconstruction
    algp.updateBath = 0;
    [thetak(:,:,jj),~,~,fk(:,:,:,jj),Ack,~] = kspace_hybrid_thermo_sseg(acqp,algp);

    % fully-sampled baseline subtraction temperature reconstruction with phase drift correction
    thetaBaseSub(:,:,jj) = -sum(angle(imgHot.*conj(imgLib).*repmat(exp(-1i*Ac),[1 1 Nc])).*abs(imgLib),3)./sum(abs(imgLib),3);

    % SENSE image temperature reconstruction with phase drift correction
    thetaSENSE(:,:,jj) = -sum(angle(imgSENSE(:,:,:,jj).*conj(imgLib).*repmat(exp(-1i*Ac),[1 1 Nc])).*abs(imgLib),3)./sum(abs(imgLib),3);

    % peak and mean phase in hot spot region
    tmpk = real(thetak(:,:,jj));
    tmpkcs = real(thetakcs(:,:,jj));
    tmpBaseSub = thetaBaseSub(:,:,jj);
    tmpSENSE = real(thetaSENSE(:,:,jj));

    thetak_max(jj) = max(-tmpk(hsmask));
    thetakcs_max(jj) = max(-tmpkcs(hsmask));
    thetaBaseSub_max(jj) = max(tmpBaseSub(hsmask));
    thetaSENSE_max(jj) = max(tmpSENSE(hsmask));

    thetak_mean(jj) = mean(-tmpk(hsmask));
    thetakcs_mean(jj) = mean(-tmpkcs(hsmask));
    thetaBaseSub_mean(jj) = mean(tmpBaseSub(hsmask));
    thetaSENSE_mean(jj) = mean(tmpSENSE(hsmask));

end

save braininvivoresults
return


%% plot results

te = 13*10^-3;
B0 = 3;
alpha = 0.01;
gamma = 2*pi*42.57;
ct = -1/te/B0/alpha/gamma;		% phase/temperature conversion factor

dispinds = [4,6,9,12]; 			% image dynamics to display
xinds = 36:100; yinds = 26:110;		% cropped display indices
clim = [0 20];				% min/max signal intensity to display

% display selected temperature maps 
figure; 
subplot(311)
im(-ct*[brainMask(xinds,yinds).*thetaBaseSub(xinds,yinds,dispinds(1)); brainMask(xinds,yinds).*thetaBaseSub(xinds,yinds,dispinds(2)); brainMask(xinds,yinds).*thetaBaseSub(xinds,yinds,dispinds(3)); brainMask(xinds,yinds).*thetaBaseSub(xinds,yinds,dispinds(4))],clim); colormap jet; axis off
title(sprintf('Fully sampled maps, dynamics: %s',num2str(dispinds)))
subplot(312)
im(-ct*[brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(1)); brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(2)); brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(3)); brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(4))],clim); colormap jet; axis off
title(sprintf('SENSE maps, %s, dynamics: %s',sampPattern,num2str(dispinds)))
subplot(313)
im(ct*[brainMask(xinds,yinds).*real(thetakcs(xinds,yinds,dispinds(1))); brainMask(xinds,yinds).*real(thetakcs(xinds,yinds,dispinds(2))); brainMask(xinds,yinds).*real(thetakcs(xinds,yinds,dispinds(3))); brainMask(xinds,yinds).*(thetakcs(xinds,yinds,dispinds(4)))],clim); colormap jet; axis off
title(sprintf('k-space brain/CG bath maps, %s, dynamics: %s',sampPattern,num2str(dispinds)))

% display temperature error maps
errSENSE = max(abs(-ct*repmat(brainMask,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetaSENSE))),[],3);
errkcs = max(abs(-ct*repmat(brainMask,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetakcs))),[],3);
figure; im([errSENSE(xinds,yinds) errkcs(xinds,yinds)],[0 10]); colormap jet
title(sprintf('%s: Max temp error across dynamics',sampPattern))    

% plot mean and max temperature change in hot spot
figure;
subplot(211);
plot(-ct*thetaBaseSub_mean,'k','LineWidth',2); 
hold on; plot(-ct*thetaSENSE_mean,'k--','LineWidth',2); 
plot(-ct*thetakcs_mean,':^','LineWidth',2,'Color',[0 0.7 0.9]);
title(sprintf('Mean temperature in hot spot'));
legend('fully sampled baseline subtraction',sprintf('kspace everywhere, %s',sampPattern),...
    sprintf('kspace brain/CG bath, %s',sampPattern),'Location','SouthEast');
xlabel('dynamic'); ylabel('Temperature (C)');
grid on;
plot(dispinds(1),0,'ko','MarkerSize',5)
plot(dispinds(2),0,'ko','MarkerSize',5)
plot(dispinds(3),0,'ko','MarkerSize',5)
plot(dispinds(4),0,'ko','MarkerSize',5)
subplot(212);
plot(-ct*thetaBaseSub_max,'k','LineWidth',2); 
hold on; plot(-ct*thetaSENSE_max,'k--','LineWidth',2); 
plot(-ct*thetakcs_max,':^','LineWidth',2,'Color',[0 0.7 0.9]);
title(sprintf('Max temperature in hot spot'));
legend('fully sampled',sprintf('kspace everywhere, %s',sampPattern),...
   sprintf('kspace brain/CG bath, %s',sampPattern),'Location','SouthEast');
xlabel('dynamic'); ylabel('Temperature (C)');
grid on;
plot(dispinds(1),0,'ko','MarkerSize',5)
plot(dispinds(2),0,'ko','MarkerSize',5)
plot(dispinds(3),0,'ko','MarkerSize',5)
plot(dispinds(4),0,'ko','MarkerSize',5)

% plot nrmse 
thetaSENSEnrmse = nrmse(-ct*thetaBaseSub,ct*thetaSENSE,repmat(brainMask,[1 1 14]));
thetaSENSEnrmse_hs = nrmse(-ct*thetaBaseSub,ct*thetaSENSE,repmat(hsmask,[1 1 14]));
thetakcsnrmse = nrmse(-ct*thetaBaseSub,ct*real(thetakcs),repmat(brainMask,[1 1 14]));
thetakcsnrmse_hs = nrmse(-ct*thetaBaseSub,ct*real(thetakcs),repmat(hsmask,[1 1 14]));

figure;
subplot(121); bar(1,thetaSENSEnrmse,0.4); grid on
set(gca,'XTickLabel',sampPattern)
hold on; bar(1,thetaSENSEnrmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2) 
ylabel('NRMSE')
xlabel('acceleration factor')
title(sprintf('NRMSE, SENSE everywhere, %s',sampPattern))
subplot(122); bar(1,thetakcsnrmse,0.4); grid on
set(gca,'XTickLabel',sampPattern)
hold on; bar(1,thetakcsnrmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2) 
ylabel('NRMSE')
xlabel('acceleration factor')
title(sprintf('NRMSE, k-space brain / CG bath, %s',sampPattern))
ylim([0 2])
legend('brain','hot spot')

