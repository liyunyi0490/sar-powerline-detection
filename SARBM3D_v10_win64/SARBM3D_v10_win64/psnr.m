function P=psnr(i1,i2)
%PSNR calculate the PSNR between two images.
% Images must be the same size.
%

if isequal(size(i1), size(i2)),
    err = sum((i1(:)-i2(:)).^2);
    err = err/numel(i1);
    if err>0,
        P = 10*log10(255^2/err);
    else
        P = inf();
    end
else
    fprintf('The images are of different sizes!\n');
	P = nan();
end
