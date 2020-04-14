;###########################################################################################
; name:		struc2arr
; date:		03.21.2001
; author:	PJW, SSAI
; old usage:	struc2arr, struc=<input structure>, array=<output array>, tags=<tag names>
; usage:	<output array> = struc2arr(<input structure>)
; purpose:	convert contents of a structure to an array
;###########################################################################################
;pro struc2arr, struc=struc, array=array, tags=tags
function struc2arr, struc

tags = tag_names(struc)
ncols = n_elements(tags)
nrows = n_elements(struc.(0)(*))
array = fltarr(ncols,nrows)

for i = 0, n_elements(tag_names(struc))-1 do array(i,*) = struc.(i)(*)

return, array
end
