;
; name:		nlines.pro
; date:		5.17.2001
; author:	PJW, SSAI
; usage:	[# lines] = nlines('input file', /all, /hdr)
; 

function nlines, file, all=all, hdr=hdr, not_seabass=not_seabass

line = ''
nline = 0L
nhdr = 0L

openr, lun, /get_lun, file

if keyword_set(not_seabass) then goto, nosb

while strmid(line,0,11) ne '/end_header' do begin
	
	readf, lun, line
	nhdr = nhdr + 1

endwhile

nosb:

while not eof(lun) do begin

	readf, lun, line
	if strtrim(line,2) ne '' then nline = nline + 1
	
endwhile

free_lun, lun

if keyword_set(hdr) then nline = nhdr
if keyword_set(all) then nline = nline + nhdr

return, nline
end
