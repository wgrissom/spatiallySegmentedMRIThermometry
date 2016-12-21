run ~/startup.m

% % manually scale waterbath signal (0, 25, 50, 75, 100%); 
% % run conventional k-space hybrid and spatially-segmented k-space hybrid method

% select the undersampling pattern to use
sampPattern = '4x,1D'; % '1x,1D', '2x,1D', '3x,1D', '4x,1D'

load('phantomdata.mat');

libInd = 2; % brain has hit steady state by second dynamic
% baseline image
imglib = img(:,:,libInd); 
mediannorm = median(abs(imglib(:)));

brainMask = imgbath(:,:,1) == 0;
bathMask = abs(imgbath(:,:,1)) > 1000;

hsmask = false(128);
hsmask(62:65,55:58) = true;

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

%% recon temperature maps
pct = [0 0.25 0.5 0.75 1]; % percentage to scale image in water bath 

for nn = 1:length(pct);
        
  for jj = libInd+1:size(img,3) % loop over dynamics
    
    % dynamic image
    %kmask = logical(kmasksv(:,:,jj));
    G = Gmri_cart(kmask,mask); % undersampled FFT operator - same as will be used in recon
    Gbath = Gmri_cart(kmask,bathMask);
    
    % define baseline and dynamic images
    imghot = img(:,:,jj)/mediannorm;
    imglib = img(:,:,libInd)/mediannorm;
    
    thetaref(:,:,jj,nn) = angle(imglib.*conj(imghot));
    
    % scale water bath in dynamic and baseline images
    imghot(bathMask) = imghot(bathMask)*pct(nn);
    imglib(bathMask) = imglib(bathMask)*pct(nn);
    L = imglib;
    dataBath = Gbath*(imghot(bathMask));
    
    
    % run k-space hybrid code
    
    % acquisition parameters
    acqp.data = G*imghot; 	% k-space data samples
    acqp.fov = fov;             % field of view
    acqp.k = kmask;		% mask of sampled k-space locations
    acqp.L = L(:);        	% baseline 'library'
    acqp.mask = mask;           % mask
    
    % algorithm parameters
    algp.dofigs = 0;            % show figures (0 -> no, 1 -> yes)
    algp.order = 1;             % polynomial order to estimate background phase drift
    algp.lam = 10^-5;     	% sparsity regularization parameter
    algp.beta = 10^-5.25;       % roughness regularization parameter
    algp.gamma = 0;             % temporal regularization parameter
    algp.modeltest = 0;         % model test
    algp.maskthresh = 0.01;     % phase shift threshold
    algp.domasked = 1;          % whether to run masked update
    algp.maskbath = bathMask; 	% mask of water bath region
    algp.maskbrain = brainMask; % mask of brain region
    algp.stopThresh = 10^-3;    % stop threshold (= fraction of previous cost that cost difference must be > than each iter)
    algp.bathPenalty = 0;	%bath roughness penalty
    algp.fBathIters = 2;	% # bath CG iters per outer iteration
    algp.thetaEps = 10^-10; 	% theta l1 penalty offset
    algp.sumMask = true;        % do a DC relaxation in the masked iterations
    algp.jointl1 = true;        % jointly sparsity-penalize the real and imaginary parts of theta
    algp.updateBath = 1;        % update image in water bath (0 -> no, 1 -> yes)
    algp.bathRecon = 'CG';      % reconstruction method in water bath ('CG','NLCG')
    algp.bathinit = zeros(dim); % initial estimate for bath image; requires algp.updateBath
    algp.thetainit = zeros(dim);% initial estimate for heat phase shift


    % spatially-segmented k-space hybrid temperature reconstruction
    [thetakcs(:,:,jj,nn),~,~,f(:,:,jj,nn),Ac(:,:,jj,nn),~] = kspace_hybrid_thermo_sseg(acqp,algp);

    % whole-image k-space hybrid temperature reconstruction
    algp.updateBath = 0;
    [thetak(:,:,jj,nn),~,~,fk(:,:,jj,nn),Ack(:,:,jj,nn),~] = kspace_hybrid_thermo_sseg(acqp,algp);

    % fully-sampled baseline subtraction temperature reconstruction with phase drift correction
    thetaBaseSub(:,:,jj,nn) = angle(imglib.*conj(imghot).*exp(1i*Ac(:,:,jj,nn)));

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

save phantomresults_2DFT
return

% figure; for jj = libInd+1:size(img,3); disp(jj); im([thetatcr(:,:,jj,nn).*brainMask;thetatcr_cgmask(:,:,jj,nn).*brainMask],[0 0.5]); colormap jet; pause; end
%% plot results
load recon2dftscalebath_3x1D_fixedkmask

TE = 0.012772;
B0 = 3;
alpha = 0.01;
gamma = 2*pi*42.57;
ct = -1/TE/B0/alpha/gamma;

jj = peakind; % display index with peak heating
xinds=30:99; yinds=15:104;
maskb = b.*abs(img(:,:,peakind))>3500;
maskb = imerode(maskb,ones(5));
hsmask = false(128);
hsmask(60:67,53:60) = true;
for nn = 1:length(pct)
    figure(41); subplot(1,length(pct),nn); im(-ct*[thetaBaseSub(xinds,yinds,jj,nn).*maskb(xinds,yinds) -real(thetak(xinds,yinds,jj,nn)).*maskb(xinds,yinds) -real(thetakcs(xinds,yinds,jj,nn)).*maskb(xinds,yinds)],[0 5]); title(sprintf('%d percent',pct(nn)*100)); colormap jet; axis off
end

for nn = 1:length(pct); 
    thetakrmse(nn) = rmse(-ct*thetaBaseSub(:,:,jj,nn),ct*real(thetak(:,:,jj,nn)),maskb)/sqrt(sum(maskb(:)));
    thetakcsrmse(nn) = rmse(-ct*thetaBaseSub(:,:,jj,nn),ct*real(thetakcs(:,:,jj,nn)),maskb)/sqrt(sum(maskb(:)));
    thetakrmse_hs(nn) = rmse(-ct*thetaBaseSub(:,:,jj,nn),ct*real(thetak(:,:,jj,nn)),hsmask)/sqrt(sum(hsmask(:)));
    thetakcsrmse_hs(nn) = rmse(-ct*thetaBaseSub(:,:,jj,nn),ct*real(thetakcs(:,:,jj,nn)),hsmask)/sqrt(sum(hsmask(:)));
end


figure(43);
subplot(121); bar(thetakrmse,0.4); grid on
set(gca,'XTickLabel',pct)
hold on; bar(thetakrmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2) 
ylabel(sprintf('RMSE (%cC)',char(176)))
xlabel('water bath image scaling')
title('RMSE, k-space everywhere')
subplot(122); bar(thetakcsrmse,0.4); grid on
set(gca,'XTickLabel',pct)
hold on; bar(thetakcsrmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2) 
ylabel(sprintf('RMSE (%cC)',char(176)))
xlabel('water bath image scaling')
title('RMSE, k-space brain / CG bath')
ylim([0 2])
legend('brain','hot spot')


for nn = 1:length(pct)
    for jj = libInd+1:size(img,3)
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
    
for nn = 1:length(pct)
    figure(nn*10);
    subplot(211);
    plot(-ct*thetaBaseSub_mean(nn,:),'k','LineWidth',2); hold on; plot(-ct*thetak_mean(nn,:),'b:','LineWidth',2); plot(-ct*thetakcs_mean(nn,:),'r-.','LineWidth',2);
    title(sprintf('Mean temperature in hot spot, %g', pct(nn)));
    legend('fully sampled baseline subtraction',sprintf('kspace everywhere, %g',pct(nn)),sprintf('kspace brain/CG bath, %g',pct(nn)));
    xlabel('dynamic'); ylabel('Temperature (C)');
    grid on;
    xlim([0 size(thetak,3)])
    subplot(212);
    plot(-ct*thetaBaseSub_max(nn,:),'k','LineWidth',2); hold on; plot(-ct*thetak_max(nn,:),'b:','LineWidth',2); plot(-ct*thetakcs_max(nn,:),'r-.','LineWidth',2);
    title(sprintf('Max temperature in hot spot, %g',pct(nn)));
    legend('fully sampled baseline subtraction',sprintf('kspace everywhere, %g',pct(nn)),sprintf('kspace brain/CG bath, %g',pct(nn)));
    xlabel('dynamic'); ylabel('Temperature (C)');
    grid on;
    xlim([0 size(thetak,3)])
end

%% plot dispinds
fnames = {'recon2dftscalebath_1x.mat', ...
    'recon2dftscalebath_2x1D_fixedkmaskindx4.mat', ...
    'recon2dftscalebath_3x1D_fixedkmask.mat', ...
    'recon2dftscalebath_4x1D_fixedkmask.mat'};
for nn = 1:length(fnames); 
    S = load(fnames{nn});
    %accfactor(nn) = S.accfactor;
    thetaBaseSub(:,:,:,nn) = S.thetaBaseSub(:,:,:,end);
    thetakcs(:,:,:,nn) = S.thetakcs(:,:,:,end);
    thetak(:,:,:,nn) = S.thetak(:,:,:,end);
    thetaBaseSub_mean(nn,:) = S.thetaBaseSub_mean(end,:);
    thetaBaseSub_max(nn,:) = S.thetaBaseSub_max(end,:);
    thetakcs_mean(nn,:) = S.thetakcs_mean(end,:);
    thetakcs_max(nn,:) = S.thetakcs_max(end,:);
    thetak_mean(nn,:) = S.thetak_mean(end,:);
    thetak_max(nn,:) = S.thetak_max(end,:);
end
thetaBase = thetaBaseSub(:,:,:,1);
thetaBase(thetaBase < 0) = 0;

% hsmask = false(128);
% hsmask(60:67,53:60) = true;
% maskb = S.brainMask.*abs(S.img(:,:,S.peakind))>3500/S.mediannorm;
% maskb = imerode(maskb,ones(5));
accfactor = [1 2 3 4];
% clear theta*nrmse*
TE = 0.012772;
B0 = 3;
alpha = 0.01;
gamma = 2*pi*42.57;
ct = -1/TE/B0/alpha/gamma;


% plot maps 
dispinds = [5,8,10,15];
figure; 
xinds=30:99; yinds=[15:104];
for nn = 1:length(fnames)
  for inds = 1:length(dispinds)
    figure(nn)
    jj = dispinds(inds);
    disp (jj)
    subplot(1,length(dispinds),inds)
    im([-ct*thetaBaseSub(xinds,yinds,jj,nn).'.*maskb(xinds,yinds).';ct*real(thetak(xinds,yinds,jj,nn)).'.*maskb(xinds,yinds).';ct*real(thetakcs(xinds,yinds,jj,nn)).'.*maskb(xinds,yinds).'].',[0 5]);
    colormap jet; axis off
    title(sprintf('%gx 2DFT: dynamic %d',accfactor(nn), dispinds(inds)))
    %pause;
  end
end

% plot max error
errk = max(abs(-ct*repmat(maskb,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetak))),[],3);
errkcs = max(abs(-ct*repmat(maskb,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetakcs))),[],3);
%figure; im([errk(xinds,yinds) errkcs(xinds,yinds)],[0 10]); colormap jet


for nn = 1:length(fnames)
    for jj = S.libInd+1:size(S.img,3)
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
    

for nn = 1:length(fnames); 
    thetakrmse(nn) = rmse(-ct*thetaBase,ct*real(thetak(:,:,:,nn)),repmat(maskb,[1 1 size(S.img,3)]))/sqrt(sum(maskb(:)*size(S.img,3)));
    thetakcsrmse(nn) = rmse(-ct*thetaBase,ct*real(thetakcs(:,:,:,nn)),repmat(maskb,[1 1 size(S.img,3)]))/sqrt(sum(maskb(:)*size(S.img,3)));
    thetakrmse_hs(nn) = rmse(-ct*thetaBase,ct*real(thetak(:,:,:,nn)),repmat(hsmask,[1 1 size(S.img,3)]))/sqrt(sum(hsmask(:)*size(S.img,3)));
    thetakcsrmse_hs(nn) = rmse(-ct*thetaBase,ct*real(thetakcs(:,:,:,nn)),repmat(hsmask,[1 1 size(S.img,3)]))/sqrt(sum(hsmask(:)*size(S.img,3)));
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


% plot fully-sampled, k-space everywhere (acc1), and all k-space brain/CG bath
figure(800);
subplot(211);
plot(-ct*thetaBaseSub_mean(1,:),'k','LineWidth',2); 
hold on; plot(-ct*thetak_mean(1,:),'k--','LineWidth',2); 
plot(-ct*thetakcs_mean(1,:),':^','LineWidth',2,'Color',[0 0.7 0.9]);
plot(-ct*thetakcs_mean(2,:),':+','LineWidth',2,'Color',[1 0.749 0]);
plot(-ct*thetakcs_mean(3,:),':s','LineWidth',2,'Color',[0.75 0.25 0]);%[0.91 0.41 0.17]);
plot(-ct*thetakcs_mean(4,:),':x','LineWidth',2,'Color',[0.6 0.4 0.8]);%[0.9 0.4 0.9]);
title(sprintf('Mean temperature in hot spot'));
% legend('fully sampled baseline subtraction',sprintf('kspace everywhere, %gx',accfactor(1)),...
%     sprintf('spatially-segmented, %gx',accfactor(1)),sprintf('spatially-segmented, %gx',accfactor(2)),...
%     sprintf('spatially-segmented, %gx',accfactor(3)),sprintf('spatially-segmented, %gx',accfactor(4)));
xlabel('dynamic'); ylabel('Temperature (C)');
grid on;
xlim([0 size(thetak,3)])
ylim([0 2])
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
plot(-ct*thetakcs_max(4,:),':x','LineWidth',2,'Color',[0.6 0.4 0.8]);%[0.9 0.4 0.9]);
title(sprintf('Max temperature in hot spot'));
legend('fully sampled',sprintf('kspace everywhere, %gx',accfactor(1)),...
    sprintf('kspace brain/CG bath, %gx',accfactor(1)),sprintf('kspace brain/CG bath, %gx',accfactor(2)),...
    sprintf('kspace brain/CG bath, %gx',accfactor(3)),sprintf('kspace brain/CG bath, %gx',accfactor(4)));
xlabel('dynamic'); ylabel('Temperature (C)');
grid on;
xlim([0 size(thetak,3)])
ylim([0 6])
plot(dispinds(1),0,'ko','MarkerSize',5)
plot(dispinds(2),0,'ko','MarkerSize',5)
plot(dispinds(3),0,'ko','MarkerSize',5)
plot(dispinds(4),0,'ko','MarkerSize',5)

