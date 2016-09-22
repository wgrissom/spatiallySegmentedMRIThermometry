run ~/startup.m

% % run conventional tcr, k-space, and new method
% % modified from quickwaterbathrecon_scalebathnoWav

load('testdata.mat','img','sampind','peakind','ktmp','fov','mask','b','imgbath','libind','dim');

img = conj(img); % complex conjugate all data so we get positive heat phase shifts

libInd = 2; % brain has hit steady state by second dynamic
% baseline image
imglib = img(:,:,libInd); 
mediannorm = median(abs(imglib(:)));
img = img/mediannorm;
imglib = img(:,:,libInd);
L = imglib;

brainMask = imgbath(:,:,1) == 0;
%bathMask = ~brainMask;
bathMask = abs(imgbath(:,:,1)) > 1000;

% set up spiral trajectory
% disp 'Synthesizing spiral k-space data'
% get a dim-shot spiral trajectory
% smax = 15000; % 150 T/m/s
% gmax = 4; % G/cm
% dt = 4e-6; % dwell time of spiral in seconds
% nshot = 24;%80; % Interleaves
% Fcoeff = [fov -fov/2 -fov/2]/12; % FOV starts at 28/12 cm; decreases to 28/24 cm.
% res = fov/dim/sqrt(2) %/12; % cm, resolution
% rmax = 1/2/res; % cm^(-1)
% [k,g,s,time,r,theta] = vds(smax,gmax,dt,nshot,Fcoeff,rmax);
% kAll = k(:)*exp(1i*2*pi/nshot*(0:nshot-1)); % duplicate to all shots
% 
% 
% d = [];
% for jj = 1:size(img,3)
%     G = {};
%     for ii = 1:nshot
%         G{ii} = Gmri([real(kAll(:,ii)) imag(kAll(:,ii))],mask,'fov',fov,'basis',{'dirac'});
%     end
%     G = block_fatrix(G,'type','col');
%     d(:,jj) = G*img(:,:,jj);%*col(sens(:,:,nn).*cplxdata(:,:,jj))
% end
% 
% 
% % reconstruct image
% jj = peakind;
% niters = 25;
% [xS,info] = qpwls_pcg(0*mask(:),G,1,d(:,jj),0,0,1,niters,mask(:));
% imgrec = embed(xS(:,end),mask);
% 
% % undersample by skipping interleaves
% 
% accfactor = 2;
% Grec = {};
% nn = 1;
% for ii = 1:accfactor:nshot;
%      Grec{nn} = Gmri([real(kAll(:,ii)) imag(kAll(:,ii))],mask,'fov',fov,'basis',{'dirac'});
%      nn = nn+1;
% end
% Grec = block_fatrix(Grec,'type','col');
% dtmp = reshape(d,[size(kAll) size(img,3)]);
% [xS,info] = qpwls_pcg(0*mask(:),Grec,1,col(dtmp(:,1:accfactor:nshot,jj)),0,0,1,niters,mask(:));
% imgrec2 = embed(xS(:,end),mask);



% set up spiral trajectory
disp 'Synthesizing spiral k-space data'
sppar.slew = 8000; % g/cm/s, grad slew rate
sppar.gmax = 4; % g/cm, max grad amp
sppar.dt = 6.4e-6;
nshot = 24;%80;

accfactor = 2;
sppar.nl = nshot/accfactor;
% [~,k,tsp,~,~,NN] = spiralgradlx6(fov/nshot,dim*sqrt(2)/nshot,sppar.dt,sppar.slew/100,sppar.gmax,1,1,1);
[~,k,tsp,~,~,NN] = spiralgradlx6(fov/sppar.nl,dim*sqrt(2)/sppar.nl,sppar.dt,sppar.slew/100,sppar.gmax,accfactor,250,50);
k = k(:);
for ii = 2:sppar.nl
    k(:,ii) = exp(1i*2*pi*(ii-1)/sppar.nl)*k(:,1);
end
k = k(1:NN(1),:);tsp = tsp(1:NN(1));

d = []; 
for jj = 1:size(img,3)
    G = {}; 
    for ii = 1:sppar.nl
        G{ii} = Gmri([real(k(:,ii)) imag(k(:,ii))],mask,'fov',fov,'basis',{'dirac'});
    end
    G = block_fatrix(G,'type','col');
    d(:,jj) = G*img(:,:,jj);%*col(sens(:,:,nn).*cplxdata(:,:,jj))
end



% % reconstruct baseline image
% jj = peakind;
% niters = 25;
% [xS,info] = qpwls_pcg(0*mask(:),G,1,d(:,jj),0,0,1,niters,mask(:));
% imgrec = embed(xS(:,end),mask);

% return
%% recon temperature maps

% skipkspacerecon = 0;
maskb = b.*abs(img(:,:,peakind))>3500/mediannorm;
maskb = imerode(maskb,ones(5));
hsmask = false(128);
hsmask(60:67,53:60) = true;

for jj = libInd+1:size(img,3) % loop over dynamics
    
    % dynamic image
    % undersample by skipping interleaves
%     Grec = {};
%     accfactor = 2;
    nn = 1;
    dtmp = reshape(d,[size(k) size(img,3)]);
%     if mod(jj,2)
%       dsamp = dtmp(:,1:accfactor:sppar.nl,jj);
%       for ii = 1:accfactor:sppar.nl
%         Grec{nn} = Gmri([real(k(:,ii)) imag(k(:,ii))],mask,'fov',fov,'basis',{'dirac'});
%         ksamp(:,:,nn) = [real(k(:,ii)) imag(k(:,ii))];
%         nn = nn+1;
%       end
%     else
%       dsamp = dtmp(:,1+(1:accfactor:sppar.nl),jj);
%       for ii = 1+(1:accfactor:sppar.nl)
%         Grec{nn} = Gmri([real(k(:,ii)) imag(k(:,ii))],mask,'fov',fov,'basis',{'dirac'});
%         ksamp(:,:,nn) = [real(k(:,ii)) imag(k(:,ii))];
%         nn = nn+1;
%       end
%     end
%     Grec = block_fatrix(Grec,'type','col');
    %[xS,info] = qpwls_pcg(0*mask(:),Grec,1,col(dtmp(:,1:accfactor:sppar.nl,jj)),0,0,1,niters,mask(:));
    %[xS,info] = qpwls_pcg(0*mask(:),G,1,dsamp(:),0,0,1,niters,mask(:));
    niters = 25;
    [xS,info] = qpwls_pcg(0*mask(:),G,1,col(dtmp(:,:,jj)),0,0,1,niters,mask(:));
    imgrec2 = embed(xS(:,end),mask);
    
    % define baseline and dynamic images
    imghot = img(:,:,jj);
    %dataBath = Gbath*(imghot(bathMask));
    
    thetaref(:,:,jj) = angle(imglib.*conj(imghot));
    
    for ii = 1:12; tmp(:,:,ii) = [real(k(:,ii)) imag(k(:,ii))]; end
    
    % run k-space hybrid code
    
    % acquisition parameters
    acqp.data = permute(dtmp(:,:,jj),[1 3 2]);% permute(dsamp,[1 3 2]);%permute(dtmp(:,1:accfactor:sppar.nl,jj),[1 3 2]);%G*imghot;% - dataBath;     	    % k-space data samples
    acqp.fov = fov;             % field of view
    acqp.k = tmp; %ksamp;%reshape(permute(ksamp,[1 3 2]),[473*12 2])%kmask;%ktmp;              % k-space sampling mask % WAG REDFLAG: This is not an actual mask, but instead is the k-space sample locs, which don't seem to be the same as the mask
    acqp.L = L(:);        	    % baseline 'library'
    acqp.mask = mask;           % mask
    %acqp.kmask = ksamp;%kmask;         % mask of sampled k-space locations
    
    % algorithm parameters
    algp.dofigs = 0;            % show figures
    algp.order = 1;             % polynomial order
    algp.lam = 10^-20;%4;%10^-4.5;%10^-5;%10^-6;% 10^-3.5];     % sparsity regularization parameter
    algp.beta = 10^-20;%5.5;%10^-20;%5;%6;%10^-5.5;%10^-5.75;%4.5;         % roughness regularization parameter
    %algp.gamma = 10^-5;%10^-3.25;         % temporal regularization parameter
    algp.modeltest = 0;         % model test
    algp.maskthresh = 0.01;     % phase shift threshold
    algp.domasked = 1;          % whether to run masked update
    algp.maskbath = bathMask; % waterbath mask
    algp.maskbrain = brainMask; % mask of brain
    algp.stopThresh = 10^-3;%6;% 10^-3;    % stop threshold (= fraction of previous cost that cost difference must be > than each iter)
    algp.bathPenalty = 0;%10^-8.25;
    algp.fBathIters = 16;%20;%4;%5;%20;%10;%5;
    algp.bathEps = 10^-10;
    algp.thetaEps = 10^-10;
    algp.bathWavPenalty = 0;%%%%%%%%%%%%%%%%%%%%%% 1e-6;%3e-7;% 10^-7;
    algp.sumMask = true; % do a DC relaxation in the masked iterations
    algp.jointl1 = true; % jointly sparsity-penalize the real and imaginary parts of theta
    
    % cs masked k-space hybrid recon
    % bathinit = zeros(dim);algp.gamma = 0;
    %bathinit = bathMask.*img(:,:,jj-1)./mediannorm;
%     if jj == libInd + 1
%        bathinit = imglib.*bathMask;
       bathinit = zeros(dim);
       thetainit = zeros(dim);
       %algp.bathPenalty = 0;
       algp.gamma = 0;
%        thetainit = zeros(dim); 
%        imginit = imglib; % set previous image for tcr (non-segmented recon)
%        imginit_cgmask = imglib;
%        imginit_brain = imglib; % set previous image for tcr (segmented recon)
     %if jj == libInd + 1
     %  thetakcs(:,:,jj-1) = zeros(dim);
     %  thetak(:,:,jj-1) = zeros(dim);
     %end
%     else
%        imginit = tcrecon(:,:,jj-1); % set previous image for tcr (non-segmented recon)
%        imginit_cgmask = tcrecon_cgmask(:,:,jj-1);
%        if ~skipkspacerecon 
%        bathinit = bathMask.*f(:,:,jj-1);
%        thetainit = angle(imglib*tcrdnorm.*conj(tcrecon_brain(:,:,jj-1,nn)));
%        imginit_brain = tcrecon_brain(:,:,jj-1,nn)*tcrdnorm; % set previous image for tcr (segmented recon)
%        end
%     end
    %if nn==1; algp.bathWavPenalty = algp.bathWavPenalty*10; end;
%     if ~skipkspacerecon
    %[thetakcs(:,:,jj),~,~,f(:,:,jj),Ac(:,:,jj),~] = kspace_hybrid_thermo_mask_v3(acqp,thetakcs(:,:,jj-1),bathinit,algp);
    [thetakcs(:,:,jj),~,~,f(:,:,jj),Ac(:,:,jj),~] = kspace_hybrid_thermo_mask_v3(acqp,thetainit,bathinit,algp);
    
%     % k-space only 
%     algp.lam = 10^-6;
    algp.maskbrain = logical(mask); % mask of brain
    algp.maskbath = logical(0*mask);
    %[thetak(:,:,jj),~,~,fk(:,:,jj),Ack(:,:,jj),~] = kspace_hybrid_thermo_mask_v3(acqp,thetak(:,:,jj-1),bathinit,algp);
    [thetak(:,:,jj),~,~,fk(:,:,jj),Ack(:,:,jj),~] = kspace_hybrid_thermo_mask_v3(acqp,thetainit,bathinit,algp);
    
    % phase difference
    thetaBaseSub(:,:,jj) = angle(imglib.*conj(imghot).*exp(1i*Ac(:,:,jj)));
%     end
    
    
    
     % plot peak and mean phase in hot spot region
     tmpkcs = real(thetakcs(:,:,jj,nn));
     tmpk = real(thetak(:,:,jj,nn));
     tmpBaseSub = thetaBaseSub(:,:,jj,nn);
     thetakcs_max(jj) = max(-tmpkcs(hsmask));
     thetak_max(jj) = max(-tmpk(hsmask));
     thetaBaseSub_max(jj) = max(tmpBaseSub(hsmask));
     thetakcs_mean(jj) = mean(-tmpkcs(hsmask));
     thetak_mean(jj) = mean(-tmpk(hsmask));
     thetaBaseSub_mean(jj) = mean(tmpBaseSub(hsmask));
%     
%     % compare to baseline subtraction with same field drift correction
%     figure; subplot(121), im([brainMask.'.*thetaBaseSub(:,:,jj,nn).' brainMask.'.*-real(thetak(:,:,jj,nn)).' brainMask.'.*-real(thetakcs(:,:,jj,nn)).'].',[0 0.5]);
%     subplot(122),plot([thetaBaseSub_max(nn) thetak_max(nn) thetakcs_max(nn) thetaBaseSub_mean(nn) thetak_mean(nn) thetakcs_mean(nn)]),axis square;
%     legend('Base sub, max','k, max','kseg, max','Base sub, mean','k, mean','kseg, mean');
%     drawnow
    



  
end

save reconspiral_scalebathnoWav_noinit

exit
return

% % figure; for jj = libInd+1:size(img,3); disp(jj); im([thetatcr(:,:,jj,nn).*brainMask;thetatcr_cgmask(:,:,jj,nn).*brainMask],[0 0.5]); colormap jet; pause; end

%% plot results
load reconspiral_scalebathnoWav_noinit


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
    im([-ct*thetaBaseSub(xinds,yinds,jj).'.*maskb(xinds,yinds).';ct*real(thetak(xinds,yinds,jj)).'.*maskb(xinds,yinds).';ct*real(thetakcs(xinds,yinds,jj)).'.*maskb(xinds,yinds).'].',[0 5]);
    colormap jet; axis off
    title(sprintf('2x spiral: dynamic %d',dispinds(inds)))
    %pause;
end

% plot error
errk = max(abs(-ct*repmat(maskb,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetak))),[],3);
errkcs = max(abs(-ct*repmat(maskb,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetakcs))),[],3);
figure; im([errk(xinds,yinds) errkcs(xinds,yinds)],[0 10]); colormap jet

thetaknrmse(nn) = nrmse(-ct*thetaBaseSub(:,:,jj,nn),ct*real(thetak(:,:,jj,nn)),maskb);
thetakcsnrmse(nn) = nrmse(-ct*thetaBaseSub(:,:,jj,nn),ct*real(thetakcs(:,:,jj,nn)),maskb);
thetaknrmse_hs(nn) = nrmse(-ct*thetaBaseSub(:,:,jj,nn),ct*real(thetak(:,:,jj,nn)),hsmask);
thetakcsnrmse_hs(nn) = nrmse(-ct*thetaBaseSub(:,:,jj,nn),ct*real(thetakcs(:,:,jj,nn)),hsmask);

 
 
figure(43);
subplot(121); bar(nn,thetaknrmse,0.4); grid on
% set(gca,'XTickLabel',pct)
hold on; bar(nn,thetaknrmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2) 
ylabel(sprintf('NRMSE (%cC)',char(176)))
% xlabel('water bath image scaling')
title('NRMSE, k-space everywhere')
subplot(122); bar(nn,thetakcsnrmse,0.4); grid on
% set(gca,'XTickLabel',pct)
hold on; bar(nn,thetakcsnrmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2) 
ylabel(sprintf('NRMSE (%cC)',char(176)))
% xlabel('water bath image scaling')
title('NRMSE, k-space brain / CG bath')
ylim([0 2])
legend('brain','hot spot')


figure;
subplot(211);
plot(-ct*thetaBaseSub_mean,'k','LineWidth',2); hold on; plot(-ct*thetak_mean,':','LineWidth',2,'Color',[0.1 0.5 0.2]); plot(-ct*thetakcs_mean,'r-.','LineWidth',2); 
title('Mean temperature in hot spot'); 
legend('fully sampled baseline subtraction','kspace everywhere, 2x','kspace brain/CG bath, 2x');
xlabel('dynamic'); ylabel('Temperature (C)'); 
grid on;
xlim([0 size(img,3)])

subplot(212);
plot(-ct*thetaBaseSub_max,'k','LineWidth',2); hold on; plot(-ct*thetak_max,':','LineWidth',2,'Color',[0.1 0.5 0.2]); plot(-ct*thetakcs_max,'r-.','LineWidth',2); 
title('Max temperature in hot spot'); 
legend('fully sampled baseline subtraction','kspace everywhere, 2x','kspace brain/CG bath, 2x');
xlabel('dynamic'); ylabel('Temperature (C)'); 
grid on;
xlim([0 size(img,3)])

    
plot(dispinds(1),0,'ko','MarkerSize',5)
plot(dispinds(2),0,'ko','MarkerSize',5)
plot(dispinds(3),0,'ko','MarkerSize',5)
plot(dispinds(4),0,'ko','MarkerSize',5)
