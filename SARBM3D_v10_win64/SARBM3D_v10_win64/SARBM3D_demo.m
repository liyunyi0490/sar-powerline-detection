close all; clear all; clc;
%SARBM3D Demo

fp = fopen('napoli.raw','rb'); 
X  = fread(fp,[256,256],'float')'; 
fclose(fp);
fp = fopen('napoli_noisy.raw','rb'); 
Z  = fread(fp,[256,256],'float')'; 
fclose(fp);

L = 1;
tic;
Y = SARBM3D_v10(Z,L);
toc,

figure;
subplot(1,3,1); imshow(X,[0 255]);
title('clean');
subplot(1,3,2); imshow(Z,[0 255]);
title(['noisy PSNR=', num2str(psnr(X,Z))]);
subplot(1,3,3); imshow(Y,[0 255]); 
title(['filtered PSNR=', num2str(psnr(X,Y))]);