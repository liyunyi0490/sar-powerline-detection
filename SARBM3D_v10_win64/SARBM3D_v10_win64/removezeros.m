function z = removezeros(z)
%REMOVEZEROS remove the zeros in the image. 
%   Replace the zeros with the fourth part of the non-zero smallest value.
%
%   y = removezeros(x)
%
%       ARGUMENT DESCRIPTION:
%               x  - image
%
%       OUTPUT DESCRIPTION:
%               y  - image without zeros
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

error('Mex file removezeros not found\n');