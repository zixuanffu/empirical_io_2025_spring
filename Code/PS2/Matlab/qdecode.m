function [out1] = qdecode(code,dtable)
% This procedure does a quick decode of a previously encoded number into
% a weakly descending n-tuple. Decoding is done using a table lookup. Each
% column of the table consists of an n-tuple; the ith column is the ith
% n-tuple to be decoded. The table is stored in the variable "dtable".
% global dtable
out1 = dtable(:,code);
end