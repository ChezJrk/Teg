%DECORATOR% %OUTPUT_TYPE% %NAME%(%CALL_LIST%){
    %TEG_VAR_TYPE% %TEG_VAR%;
    %OUTPUT_TYPE% __output__ = 0;
    %TEG_VAR_TYPE% __step__ = ((%TEG_VAR_TYPE%)(%UPPER_VAR% - %LOWER_VAR%)) / %NUM_SAMPLES%;
    for (unsigned int i = 0; i < %NUM_SAMPLES%; i++) {
        %TEG_VAR% = %LOWER_VAR% + __step__ * (i + (%TEG_VAR_TYPE%)(0.5));
        %CALL_FN%
        __output__ = __output__ + %CALL_OUT_VAR% * __step__;
    }
    return __output__;
}