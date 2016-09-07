% run ~/startup.m

% % manually scale waterbath signal, run conventional tcr, k-space, and
% % new method (0, 25, 50, 75, 100%); run tcr/k-space everywhere with scaled
% % baseline water bath at each level or no water bath at all

load('testdata.mat','img','sampind','peakind','ktmp','fov','mask','b','imgbath','libind','dim');

img = conj(img); % complex conjugate all data so we get positive heat phase shifts

libInd = 2; % brain has hit steady state by second dynamic
% baseline image
imglib = img(:,:,libInd); 
mediannorm = median(abs(imglib(:)));

brainMask = imgbath(:,:,1) == 0;
%bathMask = ~brainMask;
bathMask = abs(imgbath(:,:,1)) > 1000;

sampPattern = '4x,2D';%'3x,2D'; % '2x,1D', '4x,2D'

hsmask = false(128);
hsmask(62:65,55:58) = true;

genkmask = false;

if genkmask
    for jj = libInd+1:size(img,3) % loop over dynamics
        switch sampPattern
            case 'all'
                kmask = true(128);
            case '4x,unif'
                Nfull = 32;
                kmask = false(128);
                kmask(64-Nfull/2:64+Nfull/2-1,64-Nfull/2:64+Nfull/2-1) = true;
                kmask(1:2:end,1:2:end) = 1;
            case '2x,1D'
                % define or load a 1D 2x-undersampled pattern
                %Nfull = 24; % width of fully-sampled square region around DC
                kmask = false(128); kmask(:,sampind) = true; kmask = circshift(kmask,[0 -2]);%logical(vdPoisMex(128,128,128,24,1,2.4,Nfull,true,0));
            case '2x,2D'
                % define or load a 3x pattern
                Nfull = 32; % width of fully-sampled square region around DC
                kmask = logical(vdPoisMex(128,128,24,24,sqrt(2)*1,sqrt(2)*1,Nfull,false,0));
            case '3x,2D'
                % define or load a 3x pattern
                Nfull = 32; % width of fully-sampled square region around DC
                kmask = logical(vdPoisMex(128,128,24,24,sqrt(3)*1.05,sqrt(3)*1.05,Nfull,false,0));
            case '4x,2D'
                % define or load a 4x pattern
                Nfull = 32; % width of fully-sampled square region around DC
                kmask = logical(vdPoisMex(128,128,24,24,sqrt(4)*1.05,sqrt(4)*1.05,Nfull,false,1));
        end
        kmasksv(:,:,jj) = kmask;
        pause(1); % cheap hack to get random number stream to change between each kmask
    end
    save kmasks2 kmasksv
else
    load kmasks4x2d% kmasks2
end

%% recon temperature maps
pct = [0 0.25 0.5 0.75 1]; % percentage to scale image in water bath 

skipkspacerecon = 1;

% nn = 5;%1;
% jj = libInd+1;

for nn = 1:length(pct);
        
  for jj = libInd+1:size(img,3) % loop over dynamics
    
    % dynamic image
    kmask = kmasksv(:,:,jj);
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
    acqp.data = G*imghot;% - dataBath;     	    % k-space data samples
    acqp.fov = fov;             % field of view
    acqp.k = kmask;%ktmp;              % k-space sampling mask % WAG REDFLAG: This is not an actual mask, but instead is the k-space sample locs, which don't seem to be the same as the mask
    acqp.L = L(:);        	    % baseline 'library'
    acqp.mask = mask;           % mask
    acqp.kmask = kmask;         % mask of sampled k-space locations
    
    % algorithm parameters
    algp.dofigs = 1;            % show figures
    algp.order = 1;             % polynomial order
    algp.lam = 10^-5;%10^-6;% 10^-3.5];     % sparsity regularization parameter
    algp.beta = 10^-5.75;%6;%10^-5.5;%10^-5.75;%4.5;         % roughness regularization parameter
    algp.gamma = 10^-5;%10^-3.25;         % temporal regularization parameter
    algp.modeltest = 0;         % model test
    algp.maskthresh = 0.01;     % phase shift threshold
    algp.domasked = 1;          % whether to run masked update
    algp.maskbath = bathMask; % waterbath mask
    algp.maskbrain = brainMask; % mask of brain
    algp.stopThresh = 10^-3;%6;% 10^-3;    % stop threshold (= fraction of previous cost that cost difference must be > than each iter)
    % algp.bathStopThresh = 10^-3;  % bath POCS algorithm stopping threshold
    % algp.bathWavThresh = 10^-3;   % Bath image wavelet coefficient threshold
    % algp.doBathPOCS = 0;        % do bath POCS recon (projects between data consistency, mask consistency, sparse consistency)
    % algp.doBathCG = 0;
    algp.bathPenalty = 0;%10^-8.25;
    algp.fBathIters = 10;%5;
    algp.bathEps = 10^-10;
    algp.thetaEps = 10^-10;
    algp.bathWavPenalty = 0;%%%%%%%%%%%%%%%%%%%%%% 1e-6;%3e-7;% 10^-7;
    algp.sumMask = true; % do a DC relaxation in the masked iterations
    algp.jointl1 = true; % jointly sparsity-penalize the real and imaginary parts of theta
    %algp.bathInit = 'keyhole';  % 'keyhole' (a bit better) or 'zero-fill'
    
    % cs masked k-space hybrid recon
    % bathinit = zeros(dim);algp.gamma = 0;
    %bathinit = bathMask.*img(:,:,jj-1)./mediannorm;
    if jj == libInd + 1
       bathinit = imglib.*bathMask;%zeros(dim);
       %algp.bathPenalty = 0;
       algp.gamma = 0;
       thetainit = zeros(dim); 
       imginit = imglib; % set previous image for tcr (non-segmented recon)
       imginit_cgmask = imglib;
%        imginit_brain = imglib; % set previous image for tcr (segmented recon)
       thetakcs(:,:,jj-1,nn) = zeros(dim);
%        thetak(:,:,jj-1,nn) = zeros(dim);
    else
       imginit = tcrecon(:,:,jj-1,nn); % set previous image for tcr (non-segmented recon)
       imginit_cgmask = tcrecon_cgmask(:,:,jj-1,nn);
       if ~skipkspacerecon 
       bathinit = bathMask.*f(:,:,jj-1);
%        thetainit = angle(imglib*tcrdnorm.*conj(tcrecon_brain(:,:,jj-1,nn)));
%        imginit_brain = tcrecon_brain(:,:,jj-1,nn)*tcrdnorm; % set previous image for tcr (segmented recon)
       end
    end
    %if nn==1; algp.bathWavPenalty = algp.bathWavPenalty*10; end;
    if ~skipkspacerecon
    [thetakcs(:,:,jj,nn),~,~,f(:,:,jj,nn),Ac(:,:,jj,nn),~] = kspace_hybrid_thermo_mask_v3(acqp,thetakcs(:,:,jj-1,nn),bathinit,algp);
    
%     % k-space only 
%     algp.lam = 10^-6;
%     algp.maskbrain = logical(mask); % mask of brain
%     algp.maskbath = logical(0*mask);
%     [thetak(:,:,jj,nn),~,~,fk(:,:,jj,nn),Ack(:,:,jj,nn),~] = kspace_hybrid_thermo_mask_v3(acqp,thetak(:,:,jj-1,nn),bathinit,algp);
    
    % phase difference
    thetaBaseSub(:,:,jj,nn) = angle(imglib.*conj(imghot).*exp(1i*Ac(:,:,jj,nn)));
    end
    
    % TCR
    %algp.maskbath = bathMask; % waterbath mask
    %algp.maskbrain = brainMask; % mask of brain
    
    % whole image
    alpha2 = 0.00002;%0.15;
    niters = 100;
    itmp = imginit; itmp = itmp(mask);
    dtmp = [col(acqp.data);sqrt(alpha2)*itmp];
    Gdim = size(G);
    stmp = block_fatrix({G,sqrt(alpha2)*speye(Gdim(2),Gdim(2))},'type','col');
    [xS,info] = qpwls_pcg(col(zeros(dim)),stmp,1,dtmp,0,0,1,niters,mask);
    tcrecon(:,:,jj,nn) = embed(xS(:,end),mask);
    thetatcr(:,:,jj,nn) = angle(imglib.*conj(tcrecon(:,:,jj,nn)));%.*exp(1i*Ac(:,:,jj,nn)));
    
    % CG mask
    % set the temporal penalty to zero in the bath by using a mask
    %alpha2 = 0.002; 
    itmp = imginit_cgmask; itmp = itmp(mask);
    dtmp = [col(acqp.data);sqrt(alpha2)*itmp];
    Gdim = size(G);
    stmp = block_fatrix({G,sqrt(alpha2)*speye(Gdim(2),Gdim(2))},'type','col');
    [xS,info] = qpwls_pcg(col(0*algp.maskbrain),stmp,1,dtmp,0,0,1,niters,algp.maskbrain);
    tcrecon_cgmask(:,:,jj,nn) = embed(xS(:,end),mask);
    thetatcr_cgmask(:,:,jj,nn) = angle(imglib.*conj(tcrecon_cgmask(:,:,jj,nn)));
     
%     if ~skipkspacerecon
%     % brain only
%     alpha2 = 0.002;
% %     tmpMask = mask(brainMask);
%     Gbrain = Gmri_cart(kmask,brainMask);
% %     Grec = Gbrain;
% %     dbrain = acqp.data - Gbath*imglib(bathMask);
% %     itmp = imginit(brainMask);% itmp = itmp(mask);
% %     dtmp = [col(dbrain);sqrt(alpha2)*itmp];
% %     Gdim = size(Grec);
% %     stmp = block_fatrix({Grec,sqrt(alpha2)*speye(Gdim(2),Gdim(2))},'type','col');
% %     [xS,info] = qpwls_pcg(0*tmpMask,stmp,1,dtmp,0,0,1,niters,tmpMask);
% %     tcrecon_brain(:,:,jj) = embed(xS(:,end),brainMask);
% 
%     [thetatcr_brain(:,:,jj,nn),tcrecon_brain(:,:,jj,nn),tcrdnorm] = reconbathTCR(acqp,thetainit,imginit_brain,algp,G,Gbrain,Gbath,Ac(:,:,jj,nn),alpha2,niters); 
%     thetatcr_brain(:,:,jj,nn) = thetatcr_brain(:,:,jj,nn).*exp(1i*Ac(:,:,jj,nn));
%     end
    
%     % plot peak and mean phase in hot spot region
%     tmpkcs = real(thetakcs(:,:,jj,nn));
%     tmpk = real(thetak(:,:,jj,nn));
%     tmpBaseSub = thetaBaseSub(:,:,jj,nn);
%     thetakcs_max(nn) = max(-tmpkcs(hsmask));
%     thetak_max(nn) = max(-tmpk(hsmask));
%     thetaBaseSub_max(nn) = max(tmpBaseSub(hsmask));
%     thetakcs_mean(nn) = mean(-tmpkcs(hsmask));
%     thetak_mean(nn) = mean(-tmpk(hsmask));
%     thetaBaseSub_mean(nn) = mean(tmpBaseSub(hsmask));
%     
%     % compare to baseline subtraction with same field drift correction
%     figure; subplot(121), im([brainMask.'.*thetaBaseSub(:,:,jj,nn).' brainMask.'.*-real(thetak(:,:,jj,nn)).' brainMask.'.*-real(thetakcs(:,:,jj,nn)).'].',[0 0.5]);
%     subplot(122),plot([thetaBaseSub_max(nn) thetak_max(nn) thetakcs_max(nn) thetaBaseSub_mean(nn) thetak_mean(nn) thetakcs_mean(nn)]),axis square;
%     legend('Base sub, max','k, max','kseg, max','Base sub, mean','k, mean','kseg, mean');
%     drawnow
    



  end

end

% save recon2dft4x2dscalebath_noWav_cgbath_tcronly
% save recon2dft4x2dscalebath_noWav_alphapt002
% 
% exit
return

% figure; for jj = libInd+1:size(img,3); disp(jj); im([thetatcr(:,:,jj,nn).*brainMask;thetatcr_cgmask(:,:,jj,nn).*brainMask],[0 0.5]); colormap jet; pause; end
%% plot results
load recon2dft4x2dscalebath_noWav_cgbath
load('recon2dft4x2dscalebath_noWav_cgbath_tcronly','thetatcr','thetatcr_cgmask')
% for nn = 1:5; for jj = libInd+1:size(img,3); thetatcr_cgmask(:,:,jj,nn) = thetatcr_cgmask(:,:,jj,nn).*exp(1i*Ac(:,:,jj,nn)); end; end

TE = 0.012772;
B0 = 3;
alpha = 0.01;
gamma = 2*pi*42.57;
ct = -1/TE/B0/alpha/gamma;

thetaBaseSub(thetaBaseSub<0)=0;
thetatcr(thetatcr<0)=0;
thetatcr_cgmask(thetatcr_cgmask<0)=0;

jj = peakind; % display index with peak heating
xinds=30:99; yinds=15:104;
maskb = b.*abs(img(:,:,peakind))>3500;
maskb = imerode(maskb,ones(5));
hsmask = false(128);
hsmask(60:67,53:60) = true;
for nn = 1:length(pct)
    figure(41); subplot(1,length(pct),nn); im(-ct*[thetaBaseSub(xinds,yinds,jj,nn).*maskb(xinds,yinds) thetatcr(xinds,yinds,jj,nn).*maskb(xinds,yinds) thetatcr_cgmask(xinds,yinds,jj,nn).*maskb(xinds,yinds) -real(thetakcs(xinds,yinds,jj,nn)).*maskb(xinds,yinds)],[0 5]); title(sprintf('%d percent',pct(nn)*100)); colormap jet; axis off;
% figure(41); subplot(1,length(pct),nn); im(-ct*[thetaBaseSub(xinds,yinds,jj,nn).*maskb(xinds,yinds) thetatcr(xinds,yinds,jj,nn).*maskb(xinds,yinds) thetatcr_cgmask(xinds,yinds,jj,nn).*maskb(xinds,yinds) -real(thetak(xinds,yinds,jj,nn)).*maskb(xinds,yinds) -real(thetakcs(xinds,yinds,jj,nn)).*maskb(xinds,yinds)],[0 5]); title(sprintf('%d percent',pct(nn)*100)); colormap jet; axis off;
end
% for nn = 1:length(pct); figure(40); subplot(1,length(pct),nn); im(imghotscale(:,:,nn),[0 2.55]); axis off; title(sprintf('%d percent',pct(nn)*100)); end
for nn = 1:length(pct); 
    thetatcrnrmse(nn) = nrmse(-ct*thetaBaseSub(:,:,jj,nn),-ct*thetatcr(:,:,jj,nn),maskb);
    thetatcrbrainnrmse(nn) = nrmse(-ct*thetaBaseSub(:,:,jj,nn),-ct*thetatcr_cgmask(:,:,jj,nn),maskb);
    %thetaknrmse(nn) = nrmse(-ct*thetaBaseSub(:,:,jj,nn),ct*real(thetak(:,:,jj,nn)),maskb); 
    thetakcsnrmse(nn) = nrmse(-ct*thetaBaseSub(:,:,jj,nn),ct*real(thetakcs(:,:,jj,nn)),maskb);
    thetatcrnrmse_hs(nn) = nrmse(-ct*thetaBaseSub(:,:,jj,nn),-ct*thetatcr(:,:,jj,nn),hsmask);
    thetatcrbrainnrmse_hs(nn) = nrmse(-ct*thetaBaseSub(:,:,jj,nn),-ct*thetatcr_cgmask(:,:,jj,nn),hsmask);
    %thetaknrmse_hs(nn) = nrmse(-ct*thetaBaseSub(:,:,jj,nn),ct*real(thetak(:,:,jj,nn)),hsmask); 
    thetakcsnrmse_hs(nn) = nrmse(-ct*thetaBaseSub(:,:,jj,nn),ct*real(thetakcs(:,:,jj,nn)),hsmask); 
end



figure(43);
subplot(131); bar(thetatcrnrmse,0.4); grid on
set(gca,'XTickLabel',pct)
hold on; bar(thetatcrnrmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2) 
ylabel(sprintf('NRMSE (%cC)',char(176)))
xlabel('water bath image scaling')
title('NRMSE, TCR everywhere')
ylim([0 2])
subplot(132); bar(thetatcrbrainnrmse,0.4); grid on
set(gca,'XTickLabel',pct)
hold on; bar(thetatcrbrainnrmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2) 
ylabel(sprintf('NRMSE (%cC)',char(176)))
xlabel('water bath image scaling')
title('NRMSE, TCR brain / CG bath')
ylim([0 2])
% legend('brain','hot spot')
% subplot(223); bar(thetaknrmse,0.4); grid on
% set(gca,'XTickLabel',pct)
% hold on; bar(thetaknrmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2) 
% ylabel(sprintf('NRMSE (%cC)',char(176)))
% xlabel('water bath image scaling')
% title('NRMSE, k-space everywhere')
subplot(133); bar(thetakcsnrmse,0.4); grid on
set(gca,'XTickLabel',pct)
hold on; bar(thetakcsnrmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2) 
ylabel(sprintf('NRMSE (%cC)',char(176)))
xlabel('water bath image scaling')
title('NRMSE, k-space brain / CG bath')
ylim([0 2])
legend('brain','hot spot')



