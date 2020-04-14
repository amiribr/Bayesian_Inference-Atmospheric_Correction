; function get_f0.pro
; [structure of f0 and lambda] = get_f0(avg_wl=[array of wl])
; and avg_wl=[array of wl] calculated 10 nm average for each wl
; 9.15.2004, PJW SSAI

function get_f0, avg_wl=avg_wl

loadsb, file='thuillier.dat', struc=T

wl = T.wavelength
f0 = T.irradiance

wl_save = wl
f0_save = f0

if keyword_set(avg_wl) then begin

	avg_f0 = fltarr(n_elements(avg_wl)) + !values.f_nan

	for i = 0, n_elements(avg_wl)-1 do begin	
		idx = where(wl ge avg_wl[i]-5. and wl le avg_wl[i]+5.)
		if idx[0] ne -1 then avg_f0[i] = mean(f0[idx],/nan)
	endfor

	f0_save = avg_f0
	wl_save = avg_wl

endif

result = { f0:f0_save, lambda:wl_save }

return, result
end



