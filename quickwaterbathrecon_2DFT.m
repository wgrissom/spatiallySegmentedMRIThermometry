run ~/startup.m

load('testdata.mat','img','sampind','peakind','ktmp','fov','mask','b','imgbath','libind','dim');

img = conj(img); % complex conjugate all data so we get positive heat phase shifts

libInd = 2; % brain has hit steady state by second dynamic
% baseline image
imglib = img(:,:,libInd); 
mediannorm = median(abs(imglib(:)));
imglib = imglib/mediannorm;
L = imglib;

brainMask = imgbath(:,:,1) == 0;
%bathMask = ~brainMask;
bathMask = abs(imgbath(:,:,1)) > 1000;

sampPattern = '4x,2D';%'3x,2D'; % '2x,1D', '4x,2D'

thetakcs = zeros([dim dim size(img,3)]);
thetakey = zeros([dim dim size(img,3)]);
thetakonly = zeros([dim dim size(img,3)]);
thetaBaseSub = thetakcs;
f = thetakcs; 
fkey = thetakey; 
fk = thetakonly; 
thetakcs_max = zeros(size(img,3),1);
thetaBaseSub_max = zeros(size(img,3),1);
thetakcs_mean = zeros(size(img,3),1);
thetaBaseSub_mean = zeros(size(img,3),1);
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


for jj = libInd+1:size(img,3) % loop over dynamics
    
    kmask = kmasksv(:,:,jj);
    
    G = Gmri_cart(kmask,mask); % undersampled FFT operator - same as will be used in recon
    
    % dynamic image
    %jj = peakind;
    imghot = img(:,:,jj)/mediannorm;
    %imgbath = imgbath/mediannorm;
    %data = G*imghot;
    
    %bathMask = abs(imgbath(:,:,jj))*mediannorm > 1000;
    
    Gbath = Gmri_cart(kmask,bathMask);
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
    algp.lam = 10^-6;% 10^-3.5];     % sparsity regularization parameter
    algp.beta = 10^-6;%10^-5.5;%10^-5.75;%4.5;         % roughness regularization parameter
    algp.gamma = 10^-5;%10^-3.25;         % temporal regularization parameter
    algp.modeltest = 0;         % model test
    algp.maskthresh = 0.01;     % phase shift threshold
    algp.domasked = 1;          % whether to run masked update
    algp.maskbath = bathMask; % waterbath mask
    algp.maskbrain = brainMask; % mask of brain
    algp.stopThresh = 10^-3;    % stop threshold (= fraction of previous cost that cost difference must be > than each iter)
    % algp.bathStopThresh = 10^-3;  % bath POCS algorithm stopping threshold
    % algp.bathWavThresh = 10^-3;   % Bath image wavelet coefficient threshold
    % algp.doBathPOCS = 0;        % do bath POCS recon (projects between data consistency, mask consistency, sparse consistency)
    % algp.doBathCG = 0;
    algp.bathPenalty = 0;%10^-8.25;
    algp.fBathIters = 5;
    algp.bathEps = 10^-10;
    algp.thetaEps = 10^-10;
    algp.bathWavPenalty = 0;%%%%%%%%%%%%%%%%%%%%%% 10^-7;
    algp.sumMask = true; % do a DC relaxation in the masked iterations
    algp.jointl1 = true; % jointly sparsity-penalize the real and imaginary parts of theta
    %algp.bathInit = 'keyhole';  % 'keyhole' (a bit better) or 'zero-fill'
    
    % cs masked k-space hybrid recon
    bathinit = bathMask.*img(:,:,jj-1)./mediannorm;
    if jj == libInd + 1
        bathinit = zeros(dim);
        algp.bathPenalty = 0;
        algp.gamma = 0;
    else
        bathinit = bathMask.*f(:,:,jj-1);
    end
    [thetakcs(:,:,jj),~,~,f(:,:,jj),Ac,~] = kspace_hybrid_thermo_mask_v2(acqp,thetakcs(:,:,jj-1),bathinit,algp);
    
    % keyhole bath recon
    algp.bathInit = 'keyhole';
    algp.doBathPOCS = 0;
    algp.lam = 10^-4*[1 1];
    [thetakey(:,:,jj),~,~,fkey(:,:,jj),Ackey,~] = kspace_hybrid_thermo_mask_svcs_pocs(acqp,thetakey(:,:,jj-1),algp);
    
    % k-space only 
    algp.lam = 10^-6;
    algp.maskbrain = logical(mask); % mask of brain
    algp.maskbath = logical(0*mask);
    [thetakonly(:,:,jj),~,~,fk(:,:,jj),Ack,~] = kspace_hybrid_thermo_mask_svcs(acqp,thetakonly(:,:,jj-1),bathinit,algp);
% remove wavelet penalty from kspace_hybrid_thermo_mask code
    
    % tcr recon
%     thetatcr(:,:,jj) = recontcrmask(k,d,dim,fov,mask,G,Gbath,niters,alpha);
% 1. tcr recon within brain
% 2. nlcg recon within bath -- non-iterating?
    
    thetaBaseSub(:,:,jj) = angle(imglib.*conj(imghot).*exp(1i*Ac));
    % plot peak and mean phase in hot spot region
    tmpkcs = real(thetakcs(:,:,jj));
    tmpkonly = real(thetakonly(:,:,jj));
    tmpBaseSub = thetaBaseSub(:,:,jj);
    thetakcs_max(jj) = max(-tmpkcs(hsmask));
    thetakonly_max(jj) = max(-tmpkonly(hsmask));
    thetaBaseSub_max(jj) = max(tmpBaseSub(hsmask));
    thetakcs_mean(jj) = mean(-tmpkcs(hsmask));
    thetakonly_mean(jj) = mean(-tmpkonly(hsmask));
    thetaBaseSub_mean(jj) = mean(tmpBaseSub(hsmask));
    tmpkey = real(thetakey(:,:,jj));
    thetakey_max(jj) = max(-tmpkey(hsmask));
    thetakey_mean(jj) = mean(-tmpkey(hsmask));

    
    % compare to baseline subtraction with same field drift correction
    figure; subplot(121), im([brainMask.'.*thetaBaseSub(:,:,jj).' brainMask.'.*-real(thetakcs(:,:,jj)).'].',[0 0.5]);
    subplot(122),plot([thetaBaseSub_max thetakcs_max thetaBaseSub_mean thetakcs_mean]),axis square;
    legend('Base sub, max','kseg, max','Base sub, mean','kseg, mean');
    drawnow
    
end


%return

save recon2dft4x2d_noWavPenalty

exit



%% plot results

load recon2dft4x2d_noWavPenalty


TE = 0.012772;
B0 = 3;
alpha = 0.01;
gamma = 2*pi*42.57;
ct = -1/TE/B0/alpha/gamma;

% plot maps 
dispinds = [5,8,10,15]
figure; 
xinds=30:99; yinds=15:104;
maskb = b.*abs(img(:,:,peakind))>3500;
for inds = 1:length(dispinds)
    jj = dispinds(inds);
    disp (jj)
    subplot(1,length(dispinds),inds)
    im([-ct*thetaBaseSub(xinds,yinds,jj).'.*maskb(xinds,yinds).';ct*real(thetakonly(xinds,yinds,jj)).'.*maskb(xinds,yinds).';ct*real(thetakey(xinds,yinds,jj)).'.*maskb(xinds,yinds).';ct*real(thetakcs(xinds,yinds,jj)).'.*maskb(xinds,yinds).'].',[0 5]);
    colormap jet; axis off
    title(sprintf('4x,2d: dynamic %d',jj))
    %pause;
end

% plot error
errk = max(abs(-ct*repmat(maskb,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetakonly))),[],3);
errkey = max(abs(-ct*repmat(maskb,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetakey))),[],3);
errkcs = max(abs(-ct*repmat(maskb,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetakcs))),[],3);
figure; im([errk(xinds,yinds) errkey(xinds,yinds) errkcs(xinds,yinds)],[0 10]); colormap jet


% plot peak temp change in hot spot
figure; 
% subplot(211); 
plot(-ct*thetaBaseSub_max,'k'); hold on; plot(-ct*thetakonly_max,'g'); plot(-ct*thetakey_max,'b'); plot(-ct*thetakcs_max,'r');
title('Peak temperature in hot spot (16 voxels)'); 
legend('fully sampled baseline subtraction','kspace everywhere, 4x2D','kspace brain/keyhole bath, 4x2D','kspace brain/NLCG bath, 4x2D');
xlabel('dynamic'); ylabel('Temperature (C)'); xlim([2 27])

% plot mean temp change in hot spot
figure;
% subplot(212); 
plot(-ct*thetaBaseSub_mean,'k','LineWidth',2); hold on; 
plot(-ct*thetakonly_mean,':','LineWidth',2,'Color',[0.1 0.5 0.2])
plot(-ct*thetakey_mean,'b--','LineWidth',2); 
plot(-ct*thetakcs_mean,'r-.','LineWidth',2); 
title('Mean temperature in hot spot (16 voxels)'); 
legend('fully sampled baseline subtraction','kspace everywhere, 4x2D','kspace brain/keyhole bath, 4x2D','kspace brain/NLCG bath, 4x2D');
xlabel('dynamic'); ylabel('Temperature (C)'); xlim([2 27])

plot(dispinds(1),0,'ko','MarkerSize',5)
plot(dispinds(2),0,'ko','MarkerSize',5)
plot(dispinds(3),0,'ko','MarkerSize',5)
plot(dispinds(4),0,'ko','MarkerSize',5)

