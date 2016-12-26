function [distH, orderH] = calcHammingRank (B1, B2) % train, test
  distH = calcHammingDist(B2, B1);  % 5000*59000
  [~, orderH] = sort(distH,2); % row, ascending
  %distH(end,1:10)
end
