function [ y2, y1 ] = SARBM3D_v10( z,L )
%SARBM3Dv10 is the version 1.0 of the filter for multiplicative speckle noise
%   Denoise an image corrupted by multiplicative speckle noise
%   with the non-local filter described in "A Nonlocal SAR Image Denoising Algorithm
%   Based on LLMMSE Wavelet Shrinkage", written by
%   S. Parrilli, M. Poderico, C.V. Angelino and L. Verdoliva, IEEE Trans. on Geoscience
%   and Remote Sensing, vol. 50, no. 2, pp. 606-616, 2012.
%   Please refer to this papers for a more detailed description of
%   the algorithm.
%
%   IMA_FIL = SARBM3D_v10(IMA_NSE, L)
%
%       ARGUMENT DESCRIPTION:
%               IMA_NSE  - Noisy image (in square root intensity) 
%               L        - Number of looks of the speckle noise
%
%       OUTPUT DESCRIPTION:
%               IMA_FIL  - Fixed-point filtered image (in square root intensity)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Copyright (c) 2012 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
% All rights reserved.
% This work should only be used for nonprofit purposes.
% 
% By downloading and/or using any of these files, you implicitly agree to all the
% terms of the license, as specified in the document LICENSE.txt
% (included in this package) and online at
% http://www.grip.unina.it/download/license_closed.txt
%

    %%% Removal of zeros
    z = removezeros(z);

    %%%% First Step parameters:
    opt1.L              = L;       %% Number of looks of the speckle noise
    opt1.B_dim          = 8;       %% Number of rows/cols of block, must be a power of two
    opt1.S_dim          = 16;      %% Maximum size of the 3rd dimension of a stack, must be a power of two 
    opt1.SA_dim         = 39;      %% Diameter of search area, must be odd
    opt1.N_step         = 3;       %% Dimension of step in sliding window processing
    opt1.T2D_name       = 'daub4'; %% Transform UDWT used in the 2D spatial domain (*)
    opt1.T1D_name       = 'daub4'; %% Transform UDWT used in the 3-rd dim (*)
    opt1.tau_match      = inf();   %% Threshold for the block-distance
    opt1.beta           = 2.0;     %% Parameter of the 2D Kaiser window used in the reconstruction
    
    %%%% First Step elaboration:
    y1 = SARBM3D_step1(z,opt1);
    
    %%%% Second Step parameters:
    opt2.L              = L;       %% Number of looks of the speckle noise
    opt2.B_dim          = 8;       %% Number of rows/cols of block, must be a power of two
    opt2.S_dim          = 32;      %% Maximum size of the 3rd dimension of a stack, must be a power of two 
    opt2.SA_dim         = 39;      %% Diameter of search area, must be odd
    opt2.N_step         = 3;       %% Dimension of step in sliding window processing
    opt2.T2D_name       = 'dct';   %% Transform used in the 2D spatial domain (**)
    opt2.tau_match      = inf();   %% Threshold for the block-distance
    opt2.beta           = 2.0;     %% Parameter of the 2D Kaiser window used in the reconstruction
    
    %%%% Second Step elaboration:
    y2 = SARBM3D_step2(z,y1,opt2);

    %
    % (*) possible UDWT transforms are:
    %         'daub2'  , 'daub3'  , 'daub4',
    %         'bior1.3', 'bior1.5', 'haar' 
    %
    %
    % (**) possible transforms are:
    %         'eye'    , 'dct'    , 'haar' ,
    %         'daub2'  , 'daub3'  , 'daub4',
    %         'bior1.3', 'bior1.5'
    %
end

