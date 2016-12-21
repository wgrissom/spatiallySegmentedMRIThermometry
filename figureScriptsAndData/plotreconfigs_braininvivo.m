
fnames = {'braininvivosrs10_1x1D.mat','braininvivosrs10_2x1D_1_breakOutdiv10.mat',...
     'braininvivosrs10_3x1D_1_breakOutdiv10.mat','braininvivosrs10_4x1D_1_breakOutdiv10.mat'}

load(fnames{1})
thetaBase = thetaBaseSub;
thetaBase(thetaBase < 0) = 0;

te = 12.7720*10^-3; %tr = 27.62 * 10^-3;
B0 = 3;
alpha = 0.01;
gamma = 2*pi*42.57;
ct = -1/te/B0/alpha/gamma;
clim = [0 20];

dispinds = [4,6,9,12];

for nn = 1:length(fnames)
  load(fnames{nn});
  sampPattern = sprintf('%dx,1D',nn);

  figure(nn); 

  xinds = 36:100; yinds = 26:110;
  for inds = 1:length(dispinds)
    jj = dispinds(inds);
    disp (jj)
    subplot(1,length(dispinds),inds);
    im(-ct*[brainMask(xinds,yinds).'.*thetaBase(xinds,yinds,jj).'; brainMask(xinds,yinds).'.*thetaSENSE(xinds,yinds,jj).'; brainMask(xinds,yinds).'.*-real(thetakcs(xinds,yinds,jj)).'].',clim);
    colormap jet; axis off
    title(sprintf('%s: dynamic %d',sampPattern,jj))
    %pause;
  end

  thetaSENSE(thetaSENSE < 0) = 0;
  errSENSE = max(abs(-ct*repmat(brainMask,[1 1 size(thetaBase,3)]).*(thetaBaseSub-thetaSENSE)),[],3);
  errkcs = max(abs(-ct*repmat(brainMask,[1 1 size(thetaBase,3)]).*(thetaBaseSub+real(thetakcs))),[],3);
  figure(nn*10); im([errSENSE(xinds,yinds) errkcs(xinds,yinds)],[0 10]); colormap jet
  title(sprintf('%s: Max temp error across dynamics',sampPattern))    

end

% plot maps 
figure; 
load(fnames{2});% braininvivosrs10_2x1D_kmaskaddnfull.mat
subplot(711)
im(-ct*[brainMask(xinds,yinds).*thetaBaseSub(xinds,yinds,dispinds(1)); brainMask(xinds,yinds).*thetaBaseSub(xinds,yinds,dispinds(2)); brainMask(xinds,yinds).*thetaBaseSub(xinds,yinds,dispinds(3)); brainMask(xinds,yinds).*thetaBaseSub(xinds,yinds,dispinds(4))],clim); colormap jet; axis off
title ''
subplot(712)
im(-ct*[brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(1)); brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(2)); brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(3)); brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(4))],clim); colormap jet; axis off
title ''
subplot(713)
im(ct*[brainMask(xinds,yinds).*real(thetakcs(xinds,yinds,dispinds(1))); brainMask(xinds,yinds).*real(thetakcs(xinds,yinds,dispinds(2))); brainMask(xinds,yinds).*real(thetakcs(xinds,yinds,dispinds(3))); brainMask(xinds,yinds).*(thetakcs(xinds,yinds,dispinds(4)))],clim); colormap jet; axis off
title ''

load(fnames{3});%braininvivosrs10_3x1D
subplot(714)
im(-ct*[brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(1)); brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(2)); brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(3)); brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(4))],clim); colormap jet; axis off
title ''
subplot(715)
im(ct*[brainMask(xinds,yinds).*real(thetakcs(xinds,yinds,dispinds(1))); brainMask(xinds,yinds).*real(thetakcs(xinds,yinds,dispinds(2))); brainMask(xinds,yinds).*real(thetakcs(xinds,yinds,dispinds(3))); brainMask(xinds,yinds).*(thetakcs(xinds,yinds,dispinds(4)))],clim); colormap jet; axis off
title ''

load(fnames{4}); %braininvivosrs10_4x1D
subplot(716)
im(-ct*[brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(1)); brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(2)); brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(3)); brainMask(xinds,yinds).*thetaSENSE(xinds,yinds,dispinds(4))],clim); colormap jet; axis off
title ''
subplot(717)
im(ct*[brainMask(xinds,yinds).*real(thetakcs(xinds,yinds,dispinds(1))); brainMask(xinds,yinds).*real(thetakcs(xinds,yinds,dispinds(2))); brainMask(xinds,yinds).*real(thetakcs(xinds,yinds,dispinds(3))); brainMask(xinds,yinds).*(thetakcs(xinds,yinds,dispinds(4)))],clim); colormap jet; axis off
title ''



% plot fully-sampled, SENSE, and spatially-segmented k-space hybrid
figure(800);
subplot(211);
load(fnames{1}); %braininvivosrs10_1x1D
plot(-ct*thetaBaseSub_mean,'k','LineWidth',2); 
load(fnames{2}); %braininvivosrs10_2x1D_kmaskaddnfull.mat
hold on; plot(-ct*thetaSENSE_mean,':x','LineWidth',2,'Color',[0.9 0.4 0.9]); 
load(fnames{3}); %braininvivosrs10_3x1D
plot(-ct*thetaSENSE_mean,':*','LineWidth',2,'MarkerSize',2,'Color',[0.9 0.4 0.9]);
load(fnames{4}); %braininvivosrs10_4x1D
plot(-ct*thetaSENSE_mean,':v','LineWidth',2,'Color',[0.9 0.4 0.9]);
load(fnames{2}); %braininvivosrs10_2x1D_kmaskaddnfull.mat
plot(-ct*thetakcs_mean,':^','LineWidth',2,'Color',[0 0.7 0.9]);
load(fnames{3}); %braininvivosrs10_3x1D
plot(-ct*thetakcs_mean,':+','LineWidth',2,'Color',[1 0.749 0]);
load(fnames{4}); %braininvivosrs10_4x1D
plot(-ct*thetakcs_mean,':s','LineWidth',2,'Color',[0.75 0.25 0]);%[0.91 0.41 0.17]);
title(sprintf('Mean temperature in hot spot'));
% legend('fully sampled baseline subtraction',sprintf('kspace everywhere, %gx',accfactor(1)),...
%     sprintf('spatially-segmented, %gx',accfactor(1)),sprintf('spatially-segmented, %gx',accfactor(2)),...
%     sprintf('spatially-segmented, %gx',accfactor(3)),sprintf('spatially-segmented, %gx',accfactor(4)));
xlabel('dynamic'); ylabel('Temperature (C)');
grid on;
xlim([0 size(thetaBaseSub,3)])
plot(dispinds(1),0,'ko','MarkerSize',5)
plot(dispinds(2),0,'ko','MarkerSize',5)
plot(dispinds(3),0,'ko','MarkerSize',5)
plot(dispinds(4),0,'ko','MarkerSize',5)
subplot(212);
load(fnames{1}); 
plot(-ct*thetaBaseSub_max,'k','LineWidth',2); 
load(fnames{2});
hold on; plot(-ct*thetaSENSE_max,':+','LineWidth',2,'Color',[0.9 0.4 0.9]); 
load(fnames{3});
plot(-ct*thetaSENSE_max,':*','LineWidth',2,'MarkerSize',2,'Color',[0.9 0.4 0.9]);
load(fnames{4}); 
plot(-ct*thetaSENSE_max,':v','LineWidth',2,'Color',[0.9 0.4 0.9]);
load(fnames{2}); 
plot(-ct*thetakcs_max,':^','LineWidth',2,'Color',[0 0.7 0.9]);
load(fnames{3}); 
plot(-ct*thetakcs_max,':+','LineWidth',2,'Color',[1 0.749 0]);
load(fnames{4}); 
plot(-ct*thetakcs_max,':s','LineWidth',2,'Color',[0.75 0.25 0]);
%legend('fully sampled',sprintf('kspace everywhere, %gx',accfactor(1)),...
%    sprintf('spatially-segmented, %gx',accfactor(1)),sprintf('spatially-segmented, %gx',accfactor(2)),...
%    sprintf('spatially-segmented, %gx',accfactor(3)),sprintf('spatially-segmented, %gx',accfactor(4)));
%xlabel('dynamic'); ylabel('Temperature (C)');
grid on;
xlim([0 size(thetaBaseSub,3)])
plot(dispinds(1),0,'ko','MarkerSize',5)
plot(dispinds(2),0,'ko','MarkerSize',5)
plot(dispinds(3),0,'ko','MarkerSize',5)
plot(dispinds(4),0,'ko','MarkerSize',5)


% plot rmse
load(fnames{1});
thetaSENSE(thetaSENSE < 0) = 0;
thetaSENSErmse(1) = rmse(-ct*thetaBase,-ct*thetaSENSE,repmat(brainMask,[1 1 14]))/sqrt(sum(brainMask(:))*14);
thetaSENSErmse_hs(1) = rmse(-ct*thetaBase,-ct*thetaSENSE,repmat(hsmask,[1 1 14]))/sqrt(sum(hsmask(:))*14);
thetakcsrmse(1) = rmse(-ct*thetaBase,ct*real(thetakcs),repmat(brainMask,[1 1 14]))/sqrt(sum(brainMask(:))*14);
thetakcsrmse_hs(1) = rmse(-ct*thetaBase,ct*real(thetakcs),repmat(hsmask,[1 1 14]))/sqrt(sum(hsmask(:))*14);

load(fnames{2});
thetaSENSE(thetaSENSE < 0) = 0;
thetaSENSErmse(2) = rmse(-ct*thetaBase,-ct*thetaSENSE,repmat(brainMask,[1 1 14]))/sqrt(sum(brainMask(:))*14);
thetaSENSErmse_hs(2) = rmse(-ct*thetaBase,-ct*thetaSENSE,repmat(hsmask,[1 1 14]))/sqrt(sum(hsmask(:))*14);
thetakcsrmse(2) = rmse(-ct*thetaBase,ct*real(thetakcs),repmat(brainMask,[1 1 14]))/sqrt(sum(brainMask(:))*14);
thetakcsrmse_hs(2) = rmse(-ct*thetaBase,ct*real(thetakcs),repmat(hsmask,[1 1 14]))/sqrt(sum(hsmask(:))*14);

load(fnames{3});
thetaSENSE(thetaSENSE < 0) = 0;
thetaSENSErmse(3) = rmse(-ct*thetaBase,-ct*thetaSENSE,repmat(brainMask,[1 1 14]))/sqrt(sum(brainMask(:))*14);
thetaSENSErmse_hs(3) = rmse(-ct*thetaBase,-ct*thetaSENSE,repmat(hsmask,[1 1 14]))/sqrt(sum(hsmask(:))*14);
thetakcsrmse(3) = rmse(-ct*thetaBase,ct*real(thetakcs),repmat(brainMask,[1 1 14]))/sqrt(sum(brainMask(:))*14);
thetakcsrmse_hs(3) = rmse(-ct*thetaBase,ct*real(thetakcs),repmat(hsmask,[1 1 14]))/sqrt(sum(hsmask(:))*14);

load(fnames{4});
thetaSENSE(thetaSENSE < 0) = 0;
thetaSENSErmse(4) = rmse(-ct*thetaBase,-ct*thetaSENSE,repmat(brainMask,[1 1 14]))/sqrt(sum(brainMask(:))*14);
thetaSENSErmse_hs(4) = rmse(-ct*thetaBase,-ct*thetaSENSE,repmat(hsmask,[1 1 14]))/sqrt(sum(hsmask(:))*14);
thetakcsrmse(4) = rmse(-ct*thetaBase,ct*real(thetakcs),repmat(brainMask,[1 1 14]))/sqrt(sum(brainMask(:))*14);
thetakcsrmse_hs(4) = rmse(-ct*thetaBase,ct*real(thetakcs),repmat(hsmask,[1 1 14]))/sqrt(sum(hsmask(:))*14);

accfactor = [1 2 3 4];
figure(43);
subplot(121); bar(thetaSENSErmse,0.4); grid on
set(gca,'XTickLabel',accfactor)
hold on; bar(thetaSENSErmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2) 
ylabel(sprintf('RMSE (%cC)',char(176)))
xlabel('acceleration factor')
title('RMSE, SENSE everywhere')
subplot(122); bar(thetakcsrmse,0.4); grid on
set(gca,'XTickLabel',accfactor)
hold on; bar(thetakcsrmse_hs,'EdgeColor',[0.15 0.6 0.1],'FaceColor','None','LineWidth',2) 
ylabel(sprintf('RMSE (%cC)',char(176)))
xlabel('acceleration factor')
title('RMSE, k-space brain / CG bath')
legend('brain','hot spot')


