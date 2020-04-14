function morel_read, chl, solz, mud=mud, fn=fn, fp=fp

if keyword_set(mud) then file = 'mud.dat'
if keyword_set(fn)  then file = 'f.dat'
if keyword_set(fp)  then file = 'fp.dat'

if n_elements(file) eq 0 then begin
	print, ''
	print, 'Error: please indicate mud, f, or fp'
	print, ''
	return, -999
endif

loadstd, file=file, struc=S

reduce_struc, in=S, out=D, name='CHL'
dat = struc2arr(D)

dat0 = dat[*,where(S.solz eq  0.)]
dat1 = dat[*,where(S.solz eq 15.)]
dat2 = dat[*,where(S.solz eq 30.)]
dat3 = dat[*,where(S.solz eq 45.)]
dat4 = dat[*,where(S.solz eq 60.)]
dat5 = dat[*,where(S.solz eq 75.)]

w  = S.wl[uniq(S.wl, sort(S.wl))]
c = fix_loadsb_wl(tag_names(D))

nc = n_elements(c)
nw = n_elements(w)

wl = findgen(321) + 380.

if chl lt min(c) then begin
	vic = 0.0
endif else if chl ge max(c) then begin
	vic = nc - 1
endif else begin
	ic = 0
	while c[ic] le chl do ic = ic + 1
	vic = (ic - 1) + (chl - c[ic - 1]) / (c[ic] - c[ic - 1])
endelse

for k = 0, n_elements(wl)-1 do begin

	if wl[k] lt min(w) then begin
		vil = 0.0
	endif else if wl[k] ge max(w) then begin
		vil = nw - 1
	endif else begin
		il = 0
		while w[il] le wl[k] do il = il + 1
		vil = (il - 1) + (wl[k] - w[il - 1]) / (w[il] - w[il - 1]) 
	endelse
	
	if k eq 0 then all_vil = vil else all_vil = [all_vil, vil]

endfor

d00 = interpolate(dat0, replicate(vic, 321), all_vil)
d15 = interpolate(dat1, replicate(vic, 321), all_vil)
d30 = interpolate(dat2, replicate(vic, 321), all_vil)
d45 = interpolate(dat3, replicate(vic, 321), all_vil)
d60 = interpolate(dat4, replicate(vic, 321), all_vil)
d75 = interpolate(dat5, replicate(vic, 321), all_vil)

flo = d00
fhi = d75
tlo = 0.
thi = 75.

if solz lt 15. then begin
	fhi = d15
	thi = 15.
endif else if solz ge 15. and solz lt 30. then begin
	flo = d15
	fhi = d30
	tlo = 15.
	thi = 30.
endif else if solz ge 30. and solz lt 45. then begin
	flo = d30
	fhi = d45
	tlo = 30.
	thi = 45.
endif else if solz ge 45. and solz lt 60. then begin
	flo = d45
	fhi = d60
	tlo = 45.
	thi = 60.
endif else if solz ge 60. then begin
	flo = d60
	tlo = 60.
endif

wgt = (solz - tlo) / (thi - tlo)
ans = interpolate([[flo],[fhi]], findgen(321), replicate(wgt,321))

return, ans
end
