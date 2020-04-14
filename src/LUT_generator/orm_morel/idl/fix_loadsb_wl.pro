; FUNCTION fix_loadsb_wl 
; 1.8.2001 PJW SSAI
; change string tag names of a loadsb-loaded structure to floats
; example:  'LU443.1' becomes 443.1

function fix_loadsb_wl, tags
   
wl = fltarr(n_elements(tags))
   
for i = 0, n_elements(tags)-1 do begin
 	value = tags(i)
  	ii = strpos(value,'_')
   	if ii ne -1 then strput, value, '.', ii
   
	array = strsplit(strupcase(value), '[ABCDEFGHIJKLMNOPQRSTUVWXYZ]', /extract)
	wl(i) = float(array[0])
endfor
   
return, wl
end
