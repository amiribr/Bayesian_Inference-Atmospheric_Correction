;
; backscattering coefficient
; Loisel and Morel 1998, Ciotti et al. 1999, Morel and Maritorena 2001 
;

function get_mm01_bb, chl

wl = findgen(321) + 380.

; backscattering coefficient for seawater

bbw = 0.0038 * (400. / wl)^4.32

; Loisel and Morel 1998 bbp function

bp550 = 0.416 * chl^0.766

; Morel and Maritorena 2001 bbp with v from Ciotti et al. 1999

v = 0.768 * alog10(chl) - 1.
bbp = (0.002 + 0.01 * (0.5 - 0.25 * alog10(chl)) * (wl / 550.)^v) * bp550

bb = bbp + bbw

return, bb
end

;~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;
; diffuse attenuation coefficient
; Morel and Maritorena 2001 
; Kd = Kw + X chl^e
; requires file mm01_kbio.txt, which has columns for wl, kw, e, and X
;

function get_mm01_kd, chl

line = ''
data = fltarr(4,71)

openr, lun, 'mm01_kbio.txt', /get_lun
readf, lun, line

for i = 0, 70 do begin

	readf, lun, line
	data[*,i] = float(strsplit(line, ' ', /extract))

endfor

close, lun
free_lun, lun

k0 = data[3,*] * chl^data[2,*] + data[1,*]

wl = findgen(321) + 380.

kd = spline(data[0,*], k0, wl)

return, kd
end

;~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;
; mean cosine for downward flux
; A. Morel, personal communication, Summer 2006
; requires external subroutine morel_read.pro and associated data files
;

function get_morel_mud, solz, chl

mud = morel_read(chl, solz, /mud)

return, mud
end

;~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;
; f and Q
; Morel et al. 2002, appendix b
; for use with in-water data (senz = 0 and relaz = 0)
; requires morel_fq.txt
;

pro get_morel_fq, chl, solz, f, q

@morel_fq_appb.dat

wlut = [412.5, 442.5, 490., 510., 560., 620., 660.]
wl = findgen(321) + 380.
nw = n_elements(wlut)

clut = [0.03, 0.1, 0.3, 1., 3., 10.]
nc = n_elements(clut)

z = 1. - cos(solz * !dtor)

q1 = q0 + sq * z

if chl lt min(clut) then begin
	vic = 0
endif else if chl ge max(clut) then begin
	vic = nc - 1
endif else begin
	ic = 0
	while clut[ic] le chl do ic = ic + 1
	vic = (ic - 1) + (chl - clut[ic - 1]) / (clut[ic] - clut[ic - 1])
endelse

for k = 0, n_elements(wl)-1 do begin

	if wl[k] lt min(wlut) then begin
		vil = 0
	endif else if wl[k] ge max(wlut) then begin
		vil = nw - 1
	endif else begin
		il = 0
		while wlut[il] le wl[k] do il = il + 1
		vil = (il - 1) + (wl[k] - wlut[il - 1]) / (wlut[il] - wlut[il -1]) 
	endelse
	
	if k eq 0 then all_vil = vil else all_vil = [all_vil, vil]

endfor

q = ts_smooth(interpolate(q1, all_vil, replicate(vic, 321)), 25)

f = morel_read(chl, solz, /fp)

return
end

;~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;
; [rrs] = morel_rrs(chl, wvl=wvl, lwn=lwn)
;
; Rrs = trans * (f / Q) * {bb / (a + bb)}, where (a + bb) ~ mud * Kd
; based mostly on Morel and Maritorena 2001
;
; defaults to 380 to 700-nm, used wvl keyword for 11-nm averages
; use lwn keyword to derive lwn instead of rrs
;
; 9.25.2008 PJW, SSAI
;

function morel_rrs, chl, wvl=wvl, lwn=lwn


; define wavelength range
;
wl = findgen(321) + 380.

; default solar zenith angle is 30-degrees
;
solz = 30.0

; backscattering coefficient
;
bb = get_mm01_bb(chl)

; diffuse attenuation coefficient
;
kd = get_mm01_kd(chl)

; mean cosine for downward flux
;
mud = get_morel_mud(solz, chl)

; f and Q
;
get_morel_fq, chl, solz, f, q

; Morel and Maritorena 2001, Morel et al. 2007 iteration
;

u2 = mud
r0 = f * bb / (u2 * kd + bb)

for i = 0, 2 do begin

	u2 = mud * (1. - r0) / (1. + mud * r0 / 0.4)
	atot = 0.962 * kd * mud * (1. - r0 / f)
	r0 = f * bb / (atot + bb) 

endfor


; Gordon et al. 1988 transmission across the air-sea interface
;
rho = 0.021
rho_bar = 0.043
r_eu = 0.48
m = 1.34

t = ((1. - rho) * (1. - rho_bar)) / (m^2 * (1. - r_eu * r0))


rrs = t * r0 / q

; lwn instead of rrs
;
if keyword_set(lwn) then rrs = rrs * ((get_f0()).f0)[0:320]


; 11-nm averages centered on discrete lambda
;
if keyword_set(wvl) then begin

	nw  = n_elements(wvl)
	dsc = fltarr(nw) - 999.

	for i = 0, nw-1 do begin

		v = where(wl ge wvl[i]-5. and wl le wvl[i]+5.)
		if v[0] ne -1 then dsc[i] = mean(rrs[v],/nan)

	endfor
	
	rrs = dsc

endif


return, rrs
end
