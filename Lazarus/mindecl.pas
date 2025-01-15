{$N+}
Unit MINDECL;

{$MODE Delphi}

   {array declarations for prot min}
interface
Const     ndata = 110;
          mp = ndata;
          np = ndata;
          mma = ndata;

TYPE
   Rarray = Array[1..12,1..22] of real;
   glndata = ARRAY [1..ndata] OF real;
   glmma = ARRAY [1..mma] OF real;
   glmpbynp = ARRAY [1..mp,1..np] OF real;
   glnpbynp = ARRAY [1..np,1..np] OF real;
   glnparray = glmma;
   glmparray = glndata;
   glcvm = glmpbynp;
implementation
end.
