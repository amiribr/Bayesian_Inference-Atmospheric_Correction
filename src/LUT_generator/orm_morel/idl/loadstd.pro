; ######################################################################################
; name:		loadstd.pro
; date:		1.28.2003
; author:	PJW, SSAI
; usage:	loadstd, file=[input file], struc=[output structure]
; purpose:	create a structure of data from an input text file
;			basically, in the form: first line fields, rest data
;			/web indicates: first line fields, second line border, rest data
; ######################################################################################

pro loadstd, file=file, struc=struc, missing=missing, depth=depth, web=web

; find field names and number of columns
fields = ''
openr, lun, file, /get_lun
readf, lun, fields
if keyword_set(web) then begin
	border = ''
	readf, lun, border
endif

; remove commas and decimal points
while (((i = strpos(fields, ','))) ne -1) do strput, fields, ' ', i
while (((i = strpos(fields, '.'))) ne -1) do strput, fields, '_', i

; create an array of field names
fields = strsplit(strcompress(strupcase(fields)), ' ', /extract)
ncols = n_elements(fields)

; find number of lines in the data file
header = 1
if keyword_set(web) then header = 2
nrows = nlines(file, /not_seabass) - header

; read data into string array and close file
data = strarr(nrows)
while not eof(lun) do readf, lun, data
free_lun, lun

; create output structure
timeexists = 0
dateexists = 0
cmd = 'struc = {'
for i = 0, ncols-1 do begin
	
	type = 'fltarr'
	if fields(i) eq 'DATE' then type = 'lonarr'
	if fields(i) eq 'TIME' or fields(i) eq 'STATION' or fields(i) eq 'FILE' or fields(i) eq 'CRUISE' then type = 'strarr'
	if fields(i) eq 'TIME' THEN timeexists = 1
	if fields(i) eq 'DATE' THEN dateexists = 1
	if ((i ne ncols-1) OR timeexists OR dateexists) then comma = ',' else comma = ''
	column = fields(i) + ':' + type + '(' + string(nrows) + ')' + comma
	cmd = cmd + column

endfor
IF dateexists THEN BEGIN
	year = 'YEAR'+':'+'intarr'+ '(' + string(nrows) + ')' + comma
	month = 'MONTH'+':'+'intarr'+ '(' + string(nrows) + ')' + comma
	IF timeexists then comma = ',' else comma = ''
	day = 'DAY'+':'+'intarr'+ '(' + string(nrows) + ')' + comma
	cmd = cmd + year + month + day
ENDIF
IF timeexists THEN BEGIN
	hour = 'HOUR'+':'+'intarr'+ '(' + string(nrows) + ')' + comma
	minute = 'MINUTE'+':'+'intarr'+ '(' + string(nrows) + ')' + comma
	second = 'SECOND'+':'+'intarr'+ '(' + string(nrows) + ')'
	cmd = cmd + hour + minute + second
ENDIF
cmd = strcompress(cmd + '}', /remove_all)
void = execute(cmd)

; read data array into structure
for i = 0, nrows-1 do begin

	tdata = data(i)
	while (((j = strpos(tdata, ','))) ne -1) do strput, tdata, ' ', j
	line = strsplit(strtrim(strcompress(tdata),2), ' ', /extract)
	;while (((j = strpos(data(i), ','))) ne -1) do strput, data(i), ' ', j
	;line = strsplit(strtrim(strcompress(data(i)),2), ' ', /extract)

	for j = 0, ncols-1 do begin

		; don't check for missing values in station or time columns
		if ( fields(j) eq 'STATION' or fields(j) eq 'TIME' or fields(j) eq 'FILE' or fields(j) eq 'CRUISE' ) then begin
			struc.(j)(i) = line(j)
			if (fields(j) eq 'TIME') THEN BEGIN
				struc.hour(i) =  strmid(line(j),0,2)
				struc.minute(i) =  strmid(line(j),3,2)
				struc.second(i) =  strmid(line(j),6,2)
			ENDIF
			goto, next
		endif
		if (fields(j) eq 'DATE') THEN BEGIN
			struc.year(i) =  strmid(line(j),0,4)
			struc.month(i) =  strmid(line(j),4,2)
			struc.day(i) =  strmid(line(j),6,2)
		ENDIF

		; check for missing values in all other columns
		if keyword_set(missing) then begin
			struc.(j)(i) = !values.f_nan
			if line(j) ne missing then struc.(j)(i) = line(j)
		endif else struc.(j)(i) = line(j)

		next:
	endfor
endfor

; make depth and pressure increasing positive
if keyword_set(depth) then begin
	depth_idx = tag_index(fields,'DEPTH')
	if depth_idx ne -1 then struc.DEPTH = abs(struc.DEPTH)
	pressure_idx = tag_index(fields,'PRESSURE')
	if pressure_idx ne -1 then struc.PRESSURE = abs(struc.PRESSURE)
endif

end
