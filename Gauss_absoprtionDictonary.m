function [DA] = Gauss_absoprtionDictonary(wv,stdabsorp)

DA = zeros([length(wv),length(wv)]);
for i=1:size(DA,2)
    d = -normpdf(wv,wv(i),stdabsorp);
    DA(:,i) = d;
end
DA_norm = vnorms(DA,1);
meanIdx = floor(length(wv)/2);
DA = DA/DA_norm(meanIdx);
end