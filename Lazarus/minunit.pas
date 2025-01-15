{$N+}
Unit MINUNIT;

{$MODE Delphi}

  {Units for Protmin}



interface
Uses MINDECL;

Var   Nmin,Min,MinE: Rarray;

Procedure Func(X: real; Var P: glmma; mma: integer);

PROCEDURE svdvar(v: glnpbynp; ma,np: integer; w: glnparray;
       VAR cvm: glcvm; ncvm: integer);

PROCEDURE svbksb(u: glmpbynp; w: glnparray; v: glnpbynp;
       m,n,mp,np: integer; b: glmparray; VAR x: glnparray);



PROCEDURE svdfit(x,y,sig: glndata; ndata: integer; VAR a: glmma; mma: integer;
      VAR u: glmpbynp; VAR v: glnpbynp; VAR w: glnparray; mp,np:
      integer; VAR chisq: real);

PROCEDURE svdcmp(VAR a: glmpbynp; m,n,mp,np: integer;
       VAR w: glnparray; VAR v: glnpbynp);

implementation

Procedure Func(X: real; Var P: glmma; mma: integer);
Var i,j: longint;
begin
i:=round(x);
For j:=1 to mma do P[j]:=Min[i,j];
{writeln(P[1]:12:5,' ',P[2]:12:5,' ',P[3]:12:5);}


end; {Funcs}

PROCEDURE svdvar(v: glnpbynp; ma,np: integer; w: glnparray;
       VAR cvm: glcvm; ncvm: integer);
VAR
   k,j,i: integer;
   sum: real;
   wti: glnparray;
BEGIN
   FOR i := 1 to ma DO BEGIN
      wti[i] := 0.0;
      IF (w[i] <> 0.0) THEN  wti[i] := 1.0/(w[i]*w[i])
   END;
   FOR i := 1 to ma DO BEGIN
      FOR j := 1 to i DO BEGIN
         sum := 0.0;
         FOR k := 1 to ma DO BEGIN
            sum := sum+v[i,k]*v[j,k]*wti[k]
         END;
         cvm[i,j] := sum;
         cvm[j,i] := sum
      END
   END
END;

PROCEDURE svbksb(u: glmpbynp; w: glnparray; v: glnpbynp;
       m,n,mp,np: integer; b: glmparray; VAR x: glnparray);
VAR
   jj,j,i: integer;
   s: real;
   tmp: glnparray;
BEGIN
   FOR j := 1 to n DO BEGIN
      s := 0.0;
      IF (w[j] <> 0.0) THEN BEGIN
         FOR i := 1 to m DO BEGIN
            s := s+u[i,j]*b[i]
         END;
         s := s/w[j]
      END;
      tmp[j] := s
   END;
   FOR j := 1 to n DO BEGIN
      s := 0.0;
      FOR jj := 1 to n DO BEGIN
         s := s+v[j,jj]*tmp[jj];
      END;
      x[j] := s
   END
END;


PROCEDURE svdcmp(VAR a: glmpbynp; m,n,mp,np: integer;
       VAR w: glnparray; VAR v: glnpbynp);

LABEL 1,2,3;
CONST
   nmax=100;
VAR
   nm,l,k,j,its,i: integer;
   z,y,x,scale,s,h,g,f,c,anorm: real;
   rv1: ARRAY [1..nmax] OF real;
FUNCTION sign(a,b: real): real;
   BEGIN
      IF (b >= 0.0) THEN sign := abs(a) ELSE sign := -abs(a)
   END;
FUNCTION max(a,b: real): real;
   BEGIN
      IF (a > b) THEN max := a ELSE max := b
   END;
BEGIN
   g := 0.0;
   scale := 0.0;
   anorm := 0.0;
   FOR i := 1 to n DO BEGIN
      l := i+1;
      rv1[i] := scale*g;
      g := 0.0;
      s := 0.0;
      scale := 0.0;
      IF (i <= m) THEN BEGIN
         FOR k := i to m DO BEGIN
            scale := scale+abs(a[k,i])
         END;
         IF (scale <> 0.0) THEN BEGIN
            FOR k := i to m DO BEGIN
               a[k,i] := a[k,i]/scale;
               s := s+a[k,i]*a[k,i]
            END;
            f := a[i,i];
            g := -sign(sqrt(s),f);
            h := f*g-s;
            a[i,i] := f-g;
            IF (i <> n) THEN BEGIN
               FOR j := l to n DO BEGIN                                                                                                                                                     
                  s := 0.0;
                  FOR k := i to m DO BEGIN
                     s := s+a[k,i]*a[k,j]
                  END;
                  f := s/h;
                  FOR k := i to m DO BEGIN
                     a[k,j] := a[k,j]+
                        f*a[k,i]
                  END
               END
            END;
            FOR k := i to m DO BEGIN
               a[k,i] := scale*a[k,i]
            END
         END
      END;
      w[i] := scale*g;
      g := 0.0;
      s := 0.0;
      scale := 0.0;
      IF ((i <= m) AND (i <> n)) THEN BEGIN
         FOR k := l to n DO BEGIN
            scale := scale+abs(a[i,k])
         END;
         IF (scale <> 0.0) THEN BEGIN
            FOR k := l to n DO BEGIN
               a[i,k] := a[i,k]/scale;
               s := s+a[i,k]*a[i,k]
            END;
            f := a[i,l];
            g := -sign(sqrt(s),f);
            h := f*g-s;
            a[i,l] := f-g;
            FOR k := l to n DO BEGIN
               rv1[k] := a[i,k]/h
            END;
            IF (i <> m) THEN BEGIN
               FOR j := l to m DO BEGIN
                  s := 0.0;
                  FOR k := l to n DO BEGIN
                     s := s+a[j,k]*a[i,k]
                  END;
                  FOR k := l to n DO BEGIN
                     a[j,k] := a[j,k]
                        +s*rv1[k]
                  END
               END
            END;
            FOR k := l to n DO BEGIN
               a[i,k] := scale*a[i,k]
            END
         END
      END;
      anorm := max(anorm,(abs(w[i])+abs(rv1[i])))
   END;
   FOR i := n DOWNTO 1 DO BEGIN
      IF (i < n) THEN BEGIN
         IF (g <> 0.0) THEN BEGIN
            FOR j := l to n DO BEGIN
               v[j,i] := (a[i,j]/a[i,l])/g
            END;
            FOR j := l to n DO BEGIN
               s := 0.0;
               FOR k := l to n DO BEGIN
                  s := s+a[i,k]*v[k,j]
               END;
               FOR k := l to n DO BEGIN
                  v[k,j] := v[k,j]+s*v[k,i]
               END
            END
         END;
         FOR j := l to n DO BEGIN
            v[i,j] := 0.0;
            v[j,i] := 0.0
         END
      END;
      v[i,i] := 1.0;
      g := rv1[i];
      l := i
   END;
   FOR i := n DOWNTO 1 DO BEGIN
      l := i+1;
      g := w[i];
      IF (i < n) THEN BEGIN
         FOR j := l to n DO BEGIN
            a[i,j] := 0.0
         END
      END;
      IF (g <> 0.0) THEN BEGIN
         g := 1.0/g;
         IF (i <> n) THEN BEGIN
            FOR j := l to n DO BEGIN
               s := 0.0;
               FOR k := l to m DO BEGIN
                  s := s+a[k,i]*a[k,j]
               END;
               f := (s/a[i,i])*g;
               FOR k := i to m DO BEGIN
                  a[k,j] := a[k,j]+f*a[k,i]
               END
            END
         END;
         FOR j := i to m DO BEGIN
            a[j,i] := a[j,i]*g
         END
      END ELSE BEGIN
         FOR j := i to m DO BEGIN
            a[j,i] := 0.0
         END
      END;
      a[i,i] := a[i,i]+1.0
   END;
   FOR k := n DOWNTO 1 DO BEGIN
      FOR its := 1 to 30 DO BEGIN
         FOR l := k DOWNTO 1 DO BEGIN
            nm := l-1;
            IF ((abs(rv1[l])+anorm) = anorm) THEN GOTO 2;
            IF ((abs(w[nm])+anorm) = anorm) THEN GOTO 1
         END;
1:         c := 0.0;
         s := 1.0;
         FOR i := l to k DO BEGIN
            f := s*rv1[i];
            IF ((abs(f)+anorm) <> anorm) THEN BEGIN
               g := w[i];
               h := sqrt(f*f+g*g);
               w[i] := h;
               h := 1.0/h;
               c := (g*h);
               s := -(f*h);
               FOR j := 1 to m DO BEGIN
                  y := a[j,nm];
                  z := a[j,i];
                  a[j,nm] := (y*c)+(z*s);
                  a[j,i] := -(y*s)+(z*c)
               END
            END
         END;
2:         z := w[k];
         IF (l = k) THEN BEGIN
            IF (z < 0.0) THEN BEGIN
               w[k] := -z;
               FOR j := 1 to n DO BEGIN
               v[j,k] := -v[j,k]
            END
         END;
         GOTO 3
         END;
         IF (its = 30) THEN BEGIN
            writeln ('no convergence in 30 SVDCMP iterations'); readln
         END;
         x := w[l];
         nm := k-1;
         y := w[nm];
         g := rv1[nm];
         h := rv1[k];
         f := ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
         g := sqrt(f*f+1.0);
         f := ((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x;
         c := 1.0;
         s := 1.0;
         FOR j := l to nm DO BEGIN
            i := j+1;
            g := rv1[i];
            y := w[i];
            h := s*g;
            g := c*g;
            z := sqrt(f*f+h*h);
            rv1[j] := z;
            c := f/z;
            s := h/z;
            f := (x*c)+(g*s);
            g := -(x*s)+(g*c);
            h := y*s;
            y := y*c;
            FOR nm := 1 to n DO BEGIN
               x := v[nm,j];
               z := v[nm,i];
               v[nm,j] := (x*c)+(z*s);
               v[nm,i] := -(x*s)+(z*c)
            END;
            z := sqrt(f*f+h*h);
            w[j] := z;
            IF (z <> 0.0) THEN BEGIN
               z := 1.0/z;
               c := f*z;
               s := h*z
            END;
            f := (c*g)+(s*y);
            x := -(s*g)+(c*y);
            FOR nm := 1 to m DO BEGIN
               y := a[nm,j];
               z := a[nm,i];
               a[nm,j] := (y*c)+(z*s);
               a[nm,i] := -(y*s)+(z*c)
            END
         END;
         rv1[l] := 0.0;
         rv1[k] := f;
         w[k] := x
      END;
3:   END
END;

PROCEDURE svdfit(x,y,sig: glndata; ndata: integer; VAR a: glmma; mma: integer;
      VAR u: glmpbynp; VAR v: glnpbynp; VAR w: glnparray; mp,np:
      integer; VAR chisq: real);

CONST
   tol=1.0e-5;
VAR
   j,i: integer;
   wmax,tmp,thresh,sum: real;
   b: glndata;
   afunc: glmma;
BEGIN
   writeln('entering svdfit');
   FOR i := 1 to ndata DO BEGIN
      func(x[i],afunc,mma);
      tmp := 1.0/sig[i];
      FOR j := 1 to mma DO u[i,j] := afunc[j]*tmp;
      b[i] := y[i]*tmp
   END;
   svdcmp(u,ndata,mma,mp,np,w,v);
   wmax := 0.0;
   FOR j := 1 to mma DO IF (w[j] > wmax) THEN wmax := w[j];
   thresh := tol*wmax;
   FOR j := 1 to mma DO IF (w[j] < thresh) THEN w[j] := 0.0;
   svbksb(u,w,v,ndata,mma,mp,np,b,a);
   chisq := 0.0;
   FOR i := 1 to ndata DO BEGIN
      func(x[i],afunc,mma);
      sum := 0.0;
      FOR j := 1 to mma DO sum := sum+a[j]*afunc[j];
      chisq := chisq+sqr((y[i]-sum)/sig[i])
   END
END;
end.
