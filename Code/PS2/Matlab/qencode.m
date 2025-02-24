function [out1] = qencode(ntuple, etable, multfac)
% This procedure does a quick encode of any n-tuple given in weakly
% descending order. Encoding is done using a table lookup. Each
% column of the table consists of an n-tuple; the ith column is the ith
% n-tuple to be decoded. The table is stored in the variable "etable".
out1 = etable(sum(ntuple.*multfac)+1);

end