; $Id: STRUCT_COPY.pro $
;+
;	This Function Copies (Extracts) Tags from one Structure to a New Structure.
;
; SYNTAX:
;	Result = STRUCT_COPY, Struct, [TAGNAMES=Tagnames], [TAGS=Tags], [REMOVE=Remove])
; OUTPUT:
;	Structure
;   (-1 if input is incorrect)
; ARGUMENTS:
; 	Struct:	Input structure;
; KEYWORDS:
;	TAGNAMES:	The names of the tags to be COPIED from the structure
;	TAGS:		The tag numbers to be COPIED (as an alternative to providing the tagnames)
;	REMOVE:		Removes the specified tagnames or tags from the copied structure
; EXAMPLE:
; 	Make a Structure
;	STRUCT = CREATE_STRUCT('AA',0B,'BB',1L,'CC',0D)
;
;	Examples using tagnames:
;   Result = STRUCT_COPY(STRUCT,TAGNAMES='AA')   			; Extract Tag AA
;	Result = STRUCT_COPY(STRUCT,TAGNAMES=['AA','cc'] ) 		; Extract Tags AA,CC
;	Result = STRUCT_COPY(STRUCT,TAGNAMES=['CC','AA','Bb'] )	; Rearrange Tag Order
;	Result = STRUCT_COPY(STRUCT,TAGNAMES=['aaa'] ) 			; Result = -1 because no tagname 'aaa' exists in STRUCT
;	Result = STRUCT_COPY(STRUCT,TAGNAMES=['aaa','aa','zz']) ; Extract Tag AA

;	Result = STRUCT_COPY(STRUCT,TAGNAMES=['bb'],/REMOVE)  	; Remove Tag bb
;	Result = STRUCT_COPY(STRUCT,TAGNAMES=['bb','cc'],/REMOVE); Remove Tag bb and cc
;
;	Examples using Tags (numbers):
;	Result = STRUCT_COPY(STRUCT,TAGS=2)						; Extract Tag CC
;	Result = STRUCT_COPY(STRUCT,TAGS=[1,2])					; Extract Tags BB, CC
;	Result = STRUCT_COPY(STRUCT,TAGS=[1,2],/REMOVE)			; Remove Tags BB, CC
;
; CATEGORY:
;	STRUCTURES
; NOTES:
;	This routine does not alter the original Structure
;   This routine may be used to rearrange the order of tags in a structure
; 	Tagnames may be upper, lower or mixed case.

; VERSION:
;	Jan 14,2001
; HISTORY:
;	Mar 10,1999 Written by: J.E. O'Reilly, NOAA, 28 Tarzwell Drive, Narragansett, RI 02882
;-
; *************************************************************************
FUNCTION STRUCT_COPY, Struct, TAGNAMES=Tagnames, TAGS=Tags, REMOVE=Remove
  ROUTINE_NAME='STRUCT_COPY'

; ====================> Ensure Input information is correct
  sz_struct   = SIZE(Struct,/STRUCT)
  sz_tagnames = SIZE(Tagnames,/STRUCT)
  sz_tags     = SIZE(Tags,/STRUCT)
  IF sz_struct.type  NE 8 THEN BEGIN
    PRINT,'ERROR: Must provide Struct'
    RETURN, -1
  ENDIF
  IF sz_tagnames.n_elements EQ 0 AND sz_tags.n_elements EQ 0 THEN BEGIN
    PRINT,'ERROR: Must provide Tagnames OR Tags'
    RETURN, -1
  ENDIF
  IF sz_tagnames.n_elements GE 1 AND sz_tagnames.type   NE 7 THEN BEGIN
    PRINT,'ERROR: Tagnames must be String Type'
    RETURN, -1
  ENDIF

  ntags        = N_TAGS(Struct)
  nth_tag      = ntags -1L
  struct_names = TAG_NAMES(Struct)
  tag_subs     = LINDGEN(ntags)
  tag_targets  = -1L


; ====================>
; If Tagnames provided then find tag numbers from Tagnames, conserving the input order
  IF sz_tagnames.n_elements GE 1 AND sz_tags.n_elements EQ 0 THEN BEGIN
;   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    FOR nth=0L, N_ELEMENTS(tagnames)-1L DO BEGIN
      target = STRUPCASE(Tagnames(nth))
      ok = WHERE(struct_names EQ target,count)
      IF count EQ 1 THEN tag_targets=[tag_targets,(ok)]
    ENDFOR
  ENDIF

; ====================>
; If TAGS Provided, extract just the valid TAGS numbers for Struct, conserving the input order
   IF sz_tagnames.n_elements EQ 0 AND sz_tags.n_elements GE 1 THEN BEGIN
;   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     FOR nth=0L, N_ELEMENTS(tags)-1L DO BEGIN
      target = Tags(nth)
      ok = WHERE(tag_subs EQ target,count)
      IF count EQ 1 THEN tag_targets=[tag_targets,(ok)]
    ENDFOR
  ENDIF

; ====================> If no targets were found return -1L
  IF N_ELEMENTS(tag_targets) GE 2 THEN tag_targets = tag_targets(1:*) ELSE RETURN, -1

; ====================> IF KEYWORD_SET(REMOVE)
  IF KEYWORD_SET(REMOVE) THEN BEGIN
    temp = tag_subs
    FOR nth = 0L,N_ELEMENTS(tag_targets)-1L DO BEGIN
      target = tag_targets(nth)
      OK = WHERE(temp EQ target,count)
      IF count GE 1 THEN temp(ok) = -1
    ENDFOR
    OK = WHERE(temp NE -1,count)
    IF count GE 1 THEN tag_targets = temp(ok) ELSE RETURN, -1
  ENDIF

; ==================>
; Make a new structure to hold each of the requested valid tag numbers
  FOR nth = 0L, N_ELEMENTS(TAG_TARGETS)-1L DO BEGIN
    atag = tag_targets(nth)
    anam = Struct_names(atag)
    aval = Struct(0).(ATAG)
    IF nth EQ 0 THEN BEGIN
      template = CREATE_STRUCT(anam,aval)
    ENDIF ELSE BEGIN
      template = CREATE_STRUCT(template,anam,aval)
    ENDELSE
  ENDFOR

; ==================>
; Replicate the template to hold all data from the input Struct
  DATA = REPLICATE(template,N_ELEMENTS(Struct))

; ==================>
; Fill the data structure with the appropriate values from the input Struct
  FOR nth = 0L, N_ELEMENTS(TAG_TARGETS)-1L DO BEGIN
    atag = tag_targets(nth)
    anam = struct_names(ATAG)
    aval = Struct(0).(ATAG)
    data(*).(nth) = Struct(*).(atag)
  ENDFOR

  RETURN, DATA
  END; #####################  End of Routine ################################