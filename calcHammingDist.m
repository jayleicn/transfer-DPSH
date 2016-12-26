% compute the hamming distance between every pair of data points represented in each row of B1 and B2
function D = calcHammingDist (B1, B2) % test, train
  P1 = sign(B1 - 0.5); % {0,1} --> {-1, +1}, 5000*code_length
  P2 = sign(B2 - 0.5); % 59000*code_length

  R = size(P1, 2); % #code length?
  D = round((R - P1 * P2') / 2); %5000*59000

end
