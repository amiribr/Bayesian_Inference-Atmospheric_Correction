;############################################################################################################
; name:		reduce_struc
; date:		03.21.2001, 01.30.2002, 03.25.2002
; author:	PJW, SSAI
; usage:	reduce_struc, in=<input structure>, out=<output structure>, name=<search array>, ignore=<array>
; purpose:	use a subset of a given structure to create a new structure
;############################################################################################################
pro reduce_struc, in=in, out=out, name=name, ignore=ignore

; require ignore to be an array
if keyword_set(ignore) then ignore = [ignore]
name = [name]

tags = tag_names(in)

; set default name
; require name to be an array
if not keyword_set(name) then name = 'LU'
name = [name]

temp = -1

; find column numbers for desired parameter, unless ignore string found
for i = 0, n_elements(tags)-1 do begin
	for j = 0, n_elements(name)-1 do begin
	
		length = strlen(name[j])
	   	tag_name = strmid(tags[i],0,length)
   	
		found = -1
	   	if keyword_set(ignore) then for k = 0, n_elements(ignore)-1 do if strpos(tags[i],ignore[k]) ne -1 then found = 1	
		if tag_name eq name[j] and found eq -1 then temp = [temp, i]

	endfor
endfor

; create new structure
out = -999
if n_elements(temp) gt 1 then begin
   	cols = temp(1:(n_elements(temp) - 1))
   	out = struct_copy(in, tags=cols)
endif

end
