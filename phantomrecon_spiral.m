run ~/startup.m
% run conventional k-space and segmented k-space reconstructions using variable density spiral sampling
% select the undersampling pattern to use
sampPattern = '2.4x'; % '1x', '1.5x', '2x', '2.4x', '3x'

load('phantomdata.mat');

libInd = 2; % brain has hit steady state by second dynamic
% baseline image
imglib = img(:,:,libInd); 
mediannorm = median(abs(imglib(:)));
img = img/mediannorm;
imglib = img(:,:,libInd);
L = imglib;

brainMask = imgbath(:,:,1) == 0;
bathMask = abs(imgbath(:,:,1)) > 1000;

% set up spiral trajectory
disp 'Synthesizing spiral k-space data'
sppar.slew = 8000; 	% g/cm/s, grad slew rate
sppar.gmax = 4;    	% g/cm, max grad amp
sppar.dt = 6.4e-6;
nshot = 24;

switch sampPattern
  case '1x'
    % use 1x-sampling
    accfactor = 1;
    sppar.nl = nshot/accfactor;
    [~,k,tsp,~,~,NN] = spiralgradlx6(fov/nshot,dim*sqrt(2)/nshot,sppar.dt,sppar.slew/100,sppar.gmax,1,1,1);
  case '1.5x'
    % use 1.5x-undersampled pattern
    accfactor = 1.5;
    sppar.nl = nshot/accfactor;
    [~,k,tsp,~,~,NN] = spiralgradlx6(fov/sppar.nl,dim*sqrt(2)/sppar.nl,sppar.dt,sppar.slew/100,sppar.gmax,accfactor,200,50);
  case '2x'
    % use 2x-undersampled pattern
    accfactor = 2;
    sppar.nl = nshot/accfactor;
    [~,k,tsp,~,~,NN] = spiralgradlx6(fov/sppar.nl,dim*sqrt(2)/sppar.nl,sppar.dt,sppar.slew/100,sppar.gmax,accfactor,250,50);
  case '2.4x'
    % use 2.4x-undersampled pattern
    accfactor = 2.4;
    sppar.nl = nshot/accfactor;
    [~,k,tsp,~,~,NN] = spiralgradlx6(fov/sppar.nl,dim*sqrt(2)/sppar.nl,sppar.dt,sppar.slew/100,sppar.gmax,accfactor,300,50); 
  case '3x'
    % use 3x-undersampled pattern
    accfactor = 3;
    sppar.nl = nshot/accfactor;
    [~,k,tsp,~,~,NN] = spiralgradlx6(fov/sppar.nl,dim*sqrt(2)/sppar.nl,sppar.dt,sppar.slew/100,sppar.gmax,accfactor,325,50);
end

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


%% recon temperature maps

% skipkspacerecon = 0;
maskb = b.*abs(img(:,:,peakind))>3500/mediannorm;
maskb = imerode(maskb,ones(5));
hsmask = false(128);
hsmask(60:67,53:60) = true;

for jj = libInd+1:size(img,3) % loop over dynamics
    
    % dynamic image
    nn = 1;
    dtmp = reshape(d,[size(k) size(img,3)]);

    niters = 25;
    [xS,info] = qpwls_pcg(0*mask(:),G,1,col(dtmp(:,:,jj)),0,0,1,niters,mask(:));
    imgrec2 = embed(xS(:,end),mask);
    
    % define baseline and dynamic images
    imghot = img(:,:,jj);
    
    thetaref(:,:,jj) = angle(imglib.*conj(imghot));
    
    for ii = 1:sppar.nl; ktmp(:,:,ii) = [real(k(:,ii)) imag(k(:,ii))]; end
    
    % run k-space hybrid code
    
    % acquisition parameters
    acqp.data = permute(dtmp(:,:,jj),[1 3 2]);	% k-space data samples
    acqp.fov = fov;             		% field of view
    acqp.k = ktmp;              		% mask of sampled k-space locations
    acqp.L = L(:);        			% baseline 'library'
    acqp.mask = mask;           		% mask
    
    % algorithm parameters
    algp.dofigs = 0;                    	% show figures (0 -> no, 1 -> yes)
    algp.order = 1;                     	% polynomial order to estimate background phase drift
    algp.lam = 10^-20;          		% sparsity regularization parameter
    algp.beta = 10^-20;         		% roughness regularization parameter
    algp.gamma = 0;             		% temporal regularization parameter
    algp.modeltest = 0;         		% model test
    algp.maskthresh = 0.01;     		% phase shift threshold
    algp.domasked = 1;          		% whether to run masked update
    algp.maskbath = bathMask;   		% mask of water bath region
    algp.maskbrain = brainMask; 		% mask of brain region
    algp.stopThresh = 10^-3;    		% stop threshold (= fraction of previous cost that cost difference must be > than each iter)
    algp.stepThresh = 0.001;    		% threshold to break out of line search in NLCG algorithm
    algp.bathPenalty = 0;			% bath roughness penalty
    algp.fBathIters = 16;			% # bath CG iters per outer iteration
    algp.thetaEps = 10^-10;			% theta l1 penalty offset
    algp.sumMask = true;        		% do a DC relaxation in the masked iterations
    algp.jointl1 = true;        		% jointly sparsity-penalize the real and imaginary parts of theta
    algp.updateBath = 1;        		% update image in water bath (0 -> no, 1 -> yes)
    algp.bathRecon = 'CG';              	% reconstruction method in water bath ('CG','NLCG')
    algp.bathinit = zeros(dim);  		% initial estimate for bath image; requires algp.updateBath
    algp.thetainit = zeros(dim);        	% initial estimate for heat phase shift

    % spatially-segmented k-space hybrid temperature reconstruction
    [thetakcs(:,:,jj),~,~,f(:,:,jj),Ac,~] = kspace_hybrid_thermo_sseg(acqp,algp);

    % whole-image k-space hybrid temperature reconstruction
    algp.updateBath = 0;
    [thetak(:,:,jj),~,~,fk(:,:,jj),Ack,~] = kspace_hybrid_thermo_sseg(acqp,algp);

    % fully-sampled baseline subtraction temperature reconstruction with phase drift correction
    thetaBaseSub(:,:,jj) = angle(imglib.*conj(imghot).*exp(1i*Ac(:,:,jj)));

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
    
end

save phantomresults_spiral

%exit
return


%% plot results

% load in recons from different .mat files
fnames = {'reconspiral_acc1_again.mat', ...
    'reconspiral_acc1pt5_again.mat', ...
    'reconspiral_acc2_again.mat', ...
    'reconspiral_acc2pt4_again.mat', ...
    'reconspiral_acc3_again.mat'};
for nn = 1:length(fnames); 
    S = load(fnames{nn});
    accfactor(nn) = S.accfactor;
    thetaBaseSub(:,:,:,nn) = S.thetaBaseSub;
    thetakcs(:,:,:,nn) = S.thetakcs;
    thetak(:,:,:,nn) = S.thetak;
    thetaBaseSub_mean(nn,:) = S.thetaBaseSub_mean;
    thetaBaseSub_max(nn,:) = S.thetaBaseSub_max;
    thetakcs_mean(nn,:) = S.thetakcs_mean;
    thetakcs_max(nn,:) = S.thetakcs_max;
    thetak_mean(nn,:) = S.thetak_mean;
    thetak_max(nn,:) = S.thetak_max;
end
thetaBase = thetaBaseSub(:,:,:,1);
thetaBase(thetaBase < 0) = 0;

hsmask = S.hsmask;
maskb = S.maskb;

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
    title(sprintf('%gx spiral: dynamic %d',accfactor(nn), dispinds(inds)))
    %pause;
  end
end

% plot max error
errk = max(abs(-ct*repmat(maskb,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetak))),[],3);
errkcs = max(abs(-ct*repmat(maskb,[1 1 size(thetaBaseSub,3)]).*(thetaBaseSub+real(thetakcs))),[],3);
%figure; im([errk(xinds,yinds) errkcs(xinds,yinds)],[0 10]); colormap jet

% plot rmse
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

for nn = 1:length(fnames)
    figure(nn*10);
    subplot(211);
    plot(-ct*thetaBaseSub_mean(nn,:),'k','LineWidth',2); hold on; plot(-ct*thetak_mean(nn,:),':','LineWidth',2,'Color',[0.1 0.5 0.2]); plot(-ct*thetakcs_mean(nn,:),'r-.','LineWidth',2);
    title(sprintf('Mean temperature in hot spot, %gx', accfactor(nn)));
    legend('fully sampled baseline subtraction',sprintf('kspace everywhere, %gx',accfactor(nn)),sprintf('kspace brain/CG bath, %gx',accfactor(nn)));
    xlabel('dynamic'); ylabel('Temperature (C)');
    grid on;
    xlim([0 size(thetak,3)])
    plot(dispinds(1),0,'ko','MarkerSize',5)
    plot(dispinds(2),0,'ko','MarkerSize',5)
    plot(dispinds(3),0,'ko','MarkerSize',5)
    plot(dispinds(4),0,'ko','MarkerSize',5)
    subplot(212);
    plot(-ct*thetaBaseSub_max(nn,:),'k','LineWidth',2); hold on; plot(-ct*thetak_max(nn,:),':','LineWidth',2,'Color',[0.1 0.5 0.2]); plot(-ct*thetakcs_max(nn,:),'r-.','LineWidth',2);
    title(sprintf('Max temperature in hot spot, %gx',accfactor(nn)));
    legend('fully sampled baseline subtraction',sprintf('kspace everywhere, %gx',accfactor(nn)),sprintf('kspace brain/CG bath, %gx',accfactor(nn)));
    xlabel('dynamic'); ylabel('Temperature (C)');
    grid on;
    xlim([0 size(thetak,3)])
    plot(dispinds(1),0,'ko','MarkerSize',5)
    plot(dispinds(2),0,'ko','MarkerSize',5)
    plot(dispinds(3),0,'ko','MarkerSize',5)
    plot(dispinds(4),0,'ko','MarkerSize',5)
end

% plot fully-sampled, k-space everywhere (acc1), and all k-space brain/CG bath
figure(800);
subplot(211);
plot(-ct*thetaBaseSub_mean(1,:),'k','LineWidth',2); 
hold on; plot(-ct*thetak_mean(1,:),'k--','LineWidth',2); 
plot(-ct*thetakcs_mean(1,:),':^','LineWidth',2,'Color',[0 0.7 0.9]);
plot(-ct*thetakcs_mean(2,:),':+','LineWidth',2,'Color',[1 0.749 0]);
plot(-ct*thetakcs_mean(3,:),':s','LineWidth',2,'Color',[0.75 0.25 0]);%[0.91 0.41 0.17]);
plot(-ct*thetakcs_mean(4,:),':x','LineWidth',2,'Color',[0.6 0.4 0.8]);
plot(-ct*thetakcs_mean(5,:),':*','LineWidth',2,'MarkerSize',3,'Color',[0 0.4 0.2]);%[0.1 0.5 0.2]);
title(sprintf('Mean temperature in hot spot'));
% legend('fully sampled',sprintf('kspace everywhere, %gx',accfactor(1)),...
%     sprintf('spatially-segmented, %gx',accfactor(1)),sprintf('spatially-segmented, %gx',accfactor(2)),...
%     sprintf('spatially-segmented, %gx',accfactor(3)),sprintf('spatially-segmented, %gx',accfactor(4)),...
%     sprintf('spatially-segmented, %gx',accfactor(5)));
xlabel('dynamic'); ylabel('Temperature (C)');
grid on;
xlim([0 size(thetak,3)]); ylim([0 2]);
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
plot(-ct*thetakcs_max(4,:),':x','LineWidth',2,'Color',[0.6 0.4 0.8]);
plot(-ct*thetakcs_max(5,:),':*','LineWidth',2,'MarkerSize',3,'Color',[0 0.4 0.2]);%[0.15 0.45 0.15]);
title(sprintf('Max temperature in hot spot'));
legend('fully sampled',sprintf('kspace everywhere, %gx',accfactor(1)),...
    sprintf('spatially-segmented, %gx',accfactor(1)),sprintf('spatially-segmented, %gx',accfactor(2)),...
    sprintf('spatially-segmented, %gx',accfactor(3)),sprintf('spatially-segmented, %gx',accfactor(4)),...
    sprintf('spatially-segmented, %gx',accfactor(5)));
xlabel('dynamic'); ylabel('Temperature (C)');
grid on;
xlim([0 size(thetak,3)]); ylim([0 6])
plot(dispinds(1),0,'ko','MarkerSize',5)
plot(dispinds(2),0,'ko','MarkerSize',5)
plot(dispinds(3),0,'ko','MarkerSize',5)
plot(dispinds(4),0,'ko','MarkerSize',5)

