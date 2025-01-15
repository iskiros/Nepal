{$V+}
{$R+}

Unit Modmin;

{$MODE Delphi}

     {recalculates water compositions into minerals - modified for Melamchie waters - Al added 29/2/22 - note does biotite-chlorite mixing if mdat<6
      outputs 2 sigma errors now}

Interface
Uses MINDECL, MINUNIT;

Const     Mindat   = 'Min_Koto.prn';
          Waterdat = 'Traverse_34.prn';
          PlagAn   = 0.25;      {mol fraction An in plagioclase}
          frbio = 0.80; {mol fract biotite in biotite-chlorite mixture}
          Mgmult = 1.0;  {factor on chlorite Mg}
          Biomult = 1.0; {Factor on biotite Mg}

Type   farray = Array[1..11,1..11] of real;
Var    KD,XX,Water,Wat,Swat,sigA,TotC: glndata;
       {XX is mineral number, Water & Wat are rock compositions
        Srock is 1 sigma on rck, sigA ?}
       Fmin,Minin: Rarray;
       k,j,i,it,ndat,mdat,imin,mini,ntot,ict,minnum,DensI: integer;
       Ident,Isel: Array[1..ndata] of integer;
       BBe,WtBB,WtBBe,Totl,Nmol,MolA,Mmol,AA,BB,VV,EA,MinDen,MinD,MinV,Minwt,
       MwtS: glmma;
       U,cov: glmpbynp;
       V,Total: glnpbynp;
       W: glnparray;
       A,Ca,Mg,total1,chis,Nmolbio,Nmolqtz,Nmolplag,Nmolmusc,Nmolcb,
       Dist,tot,NmlKsp,Nmolilm,Camol,Mgmol,Namol,SCamol,Kmol,Mgst,Cast,
       Natot,Sitot,plgAn,Vtot: real;
       infile,inf2,out1,out2: text;
       El: Array[1..12] of string[5];
       Ell: Array[1..12] of string[5];
       Minname,MinnameE: Array[1..10] of string[9];
       Sname: Array[-1..21] of string[8];
       Samp: string[12]; Samp1: string[12];
       Asamp: Array[1..110] of string[42];
       sample,mineral,component: string[9];
       MMin: rarray;
Implementation
Begin
{caution check whether wt% or mol input and whether converts in program}
assign(infile,Waterdat);
reset(infile);
assign(inf2,Mindat);
reset(inf2);
assign(out1,'m.r');
rewrite(out1);
assign(out2,'m.rr');
rewrite(out2);
{Selected minerals}
Isel[1]:=1;Isel[2]:=2; Isel[3]:=3; Isel[4]:=4; Isel[5]:=5; Isel[6]:=6;
{water element order Na K  Ca  Mg Si Al
                     1  2   3   4  5  6  7  8}
Mmol[1]:=60.09; Mmol[2]:=79.9; Mmol[3]:=101.94;
Mmol[4]:=71.85; Mmol[5]:=40.32; Mmol[6]:=56.08;
Mmol[7]:=61.982; Mmol[8]:=94.20;
{correction from millimolar
Mmol[1]:=1000.0; Mmol[2]:=1000.0; Mmol[3]:=2000.0;
Mmol[4]:=2000.0; Mmol[5]:=1000.0; Mmol[6]:=1000.0;
Mmol[7]:=2000.0; Mmol[8]:=2000.0;}
MolA[1]:=60.09; MolA[2]:=267.82; MolA[3]:=278.34; MolA[4]:=380.25;
MolA[5]:=456.05; MolA[6]:=140.95; MolA[7]:=151.75; MolA[8]:=152.0;
MolA[9]:= 210.9; MolA[10]:=17.898;

El[1]:='Na'; El[2]:= ' K'; El[3]:='Ca'; El[4]:= 'Mg'; El[5]:='Si';
El[6]:= 'Al';



{Water element order Na K Ca Mg Si Al
                     1  2  3  4  5  6 }


{Read mineral compositions}
{Min[i,j] i is component, j is mineral
  1    2    3     4    5  6    7    8
  Min[i,j] i is element, j is mineral}

Minnum:=6; {Number of input minerals    Plag        Biotite     Calcite     Kaolinite   Smectite    Chlorite}
Minden[1]:=2.65; Minden[2]:=3.0; Minden[3]:=2.72; Minden[4]:=2.65; Minden[5]:=2.65; Minden[6]:=3.0;
Minwt[1]:=266.23; Minwt[2]:=468.56; Minwt[3]:=99.04; Minwt[4]:=290.73; Minwt[5]:=290; Minwt[6]:=691.27;


mdat:=5;  {No minerals used for decomposition - numbers set in Isel[i]}
ndat:=6; {components}
DensI:=3; {Number of minerals to sum as inputs to volume calculation}
Read(inf2,Samp1);
For j:=1 to minnum do read(inf2,MinnameE[j]);
For j:=1 to minnum do write(MinnameE[j],'  ');
readln(inf2);

For i:=1 to 6 do    {loop over elements}
    begin
    read(inf2,Component);
    For j:=1 to minnum do read(inf2,Minin[i,j]);
    readln(inf2);
    end;
{Set plagioclase An}
Minin[5,1]:=1.0-PlagAn;
Minin[4,1]:=PlagAn;
Min[1,1]:= 3*(1-PlagAn)+2*PlagAn;
Min[2,1]:= (1-PlagAn) + 2*PlagAn;
Minin[3,6]:=Mgmult*Minin[3,6];
Minin[3,2]:=Biomult*Minin[3,2];
writeln('Data read in now');
{input element order Si Al Mg Ca Na K
Output order Na K Ca Mg Si Al}
For j:= 1 to Minnum Do    {reorders elements in mineral array to those used for water deconvoluton}
        Begin
        MinE[1,j]:=Minin[5,j];
        MinE[2,j]:=Minin[6,j];
        MinE[3,j]:=Minin[4,j];
        MinE[4,j]:=Minin[3,j];
        MinE[5,j]:=Minin[1,j];
        MinE[6,j]:=Minin[2,j];
        end;
If (mdat<6) then For i:= 1 to ndat Do MinE[i,2]:=frbio*MinE[i,2]+(1-frbio)*MinE[i,6];     {Makes biotite-chlorite mixture}
write(out1,'             ');
For j:= 1 to mdat Do
                  Begin
                  MinD[j]:=MinDen[Isel[j]];
                  Minwt[j]:=Minwt[Isel[j]];
                  For i:=1 to ndat Do Min[i,j]:=MinE[i,Isel[j]];
                  end;
writeln('MinE[4,6] ',MinE[4,6]:8:3);
If (mdat<6) then Minwt[2]:= frbio*Minwt[2]+(1-frbio)*Minwt[6];
plgAn:= Min[3,1]*100.0;
For j:= 1 to mdat Do Minname[j]:=MinnameE[Isel[j]];
writeln('mdat= ',mdat:3); readln;
write(out1,'    Plag  %An = ',plgAn:5:1,'  Chl Mg mult = ',Mgmult:8:3,'  Biotite Mg mult = ',Biomult:8:3);
If (mdat<6) then writeln(out1,'  Biotite/(Biotite+chlorite) = ',frBio:8:3)
            else writeln(out1);
write(Out1,'              ');
for i:=1 to mdat do write(out1,Minname[i],'         ');
for i:=1 to DensI do write(out1,Minname[i],'');
writeln(Out1,' chi**2   Na/Si');
for i:=1 to ndat do XX[i]:=i;
readln(infile);  readln(infile);

ict:=0;

While not eof(infile) do
  begin
  ict:=ict+1;
  read(infile,Samp);
  Asamp[ict]:=Samp;
  For i:= 1 to ndat do  read(infile,Water[i]);
  readln(infile);

  If (ict>0) then Begin
                  write(Samp,' ');
                  for i:=1 to ndat do  Begin write(' i= ',i:2,' ',Water[i]:8:3,' ');   end;
                  writeln;
                  end;







{number of mols per 1000 g of rock}

  For i:= 1 to ndat do Wat[i]:=Water[i];  {not modified for mmmolar input}

  For i:= 1 to ndat do If (Wat[i]>0.5) then Swat[i]:=Wat[i]*0.05
                                        else Swat[i]:=0.1;



{Needs mineral data in order plag musc, biot, carb, ilm, chl, garnet}

  SVDFit(XX,Wat,Swat,ndat,BB,mdat,U,V,W,ndat,mdat,chis);
  writeln('finished svdfit 1');

  svdvar(V,mdat,ndat,W,cov,ndat);
  writeln('Covariance matrix ');
  For i:=1 to mdat do
    begin
    For j:=1 to mdat do  write(cov[i,j]:8:6,'   ');
    writeln;
    end;
  writeln;
  For j:=1 to mdat do BBe[j]:=sqrt(cov[j,j]);
  Tot:=0;
  {Calculate input Na/Si}
  Natot:=0.0; Sitot:=0.0;
  For j:= 1 to mdat Do
                     Begin
                     Natot:=Natot + BB[j]*Min[1,j];
                     Sitot:=Sitot + BB[j]*Min[5,j];
                     end;
  writeln('Mineral fractions and errors and input Na/Si');
  Vtot:=0.0;
  For j:= 1 to DensI Do
              Begin
              Minv[j]:=BB[j]*Minwt[j]/minD[j];
              Vtot:=Vtot+Minv[j];
              end;
  For j:=1 to DensI Do Minv[j]:=100*Minv[j]/Vtot;
  write(out1,Samp,' ');
  for i:=1 to mdat do write(out1,BB[i]:7:2,'  ',2*BBe[i]:7:2,'  ');

  for j:=1 to DensI do write(out1,Minv[j]:7:2,'  ');
  writeln(out1,'  ',Chis:6:2,'  ',Natot/Sitot:8:3,'  ');;

writeln('About to start total ',Asamp[ict]);

  {Check returns original composition}
  For j:=1 to ndat do    {loop over ndat elements}
    begin
    total[ict,j]:=0;
    For i:=1 to mdat do   {sum element j in mdat minerals i}
       Begin
       {writeln('total = ',total[ict,j]:6:3,' BB[',i:1,']= ',BB[i]:6:2,' Min[',j:1,',',i:1,']= ',Min[j,i]:6:4);}
       total[ict,j]:=total[ict,j]+BB[i]*Min[j,i];
       {writeln('total = ',total[ict,j]:6:3);}
       
       end;
    end;
  writeln(out2,' compositions  ',Samp);
Writeln(out2,'              Recovered water compositions                     Input water compositions');
Write(Out2,'            ');
For i:=1 to ndat do write(out2,'   ',El[i],'    ');
write(out2,'     ');
For i:=1 to ndat do write(out2,'   ',El[i],'    ');
writeln(out2);
Write(out2,Samp,'  ');
For i:=1 to ndat do TotC[i]:=0.0;
For i:= 1 to ndat Do  For j:=1 to mdat do TotC[i]:=TotC[i] + BB[j]*Min[i,j];
For i:=1 to ndat Do write(out2,TotC[i]:7:3,'  ');
write(Out2,'      ');
For i:=1 to ndat Do write(out2,Wat[i]:7:3,'  ');
writeln(Out2);

  end; {end cycling through data}

writeln(out1);
write(out1,'              ');
For j:= 1 to ndat do write(out1,'  ',El[j],'    ');
writeln(out1);
For i:=1 to ict do
   Begin
   Write(out1,Asamp[i],' ');
   For j:=1 to ndat do write(out1,Total[i,j]:7:2,' ');
   writeln(out1);
   end;
write(Out2,'     ');
For j:= 1 to mdat Do write(out2,' ',Minname[j],'  ');
writeln(out2);
For i:= 1 to ndat Do
         Begin
         Write(out2,El[i],'  ');
         For j:= 1 to mdat Do write(out2,Min[i,j]:7:3,'      ');
         writeln(out2);
         end;

close(infile);
close(inf2);
close(out1);
close(out2);
end.
