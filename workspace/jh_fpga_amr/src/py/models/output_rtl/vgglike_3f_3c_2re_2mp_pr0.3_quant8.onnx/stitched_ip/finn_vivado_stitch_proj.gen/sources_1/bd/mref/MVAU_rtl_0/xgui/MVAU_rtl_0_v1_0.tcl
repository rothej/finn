# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "ACCU_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ACTIVATION_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "COMPUTE_CORE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FORCE_BEHAVIORAL" -parent ${Page_0}
  ipgui::add_param $IPINST -name "INPUT_STREAM_WIDTH_BA" -parent ${Page_0}
  ipgui::add_param $IPINST -name "IS_MVU" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MW" -parent ${Page_0}
  ipgui::add_param $IPINST -name "NARROW_WEIGHTS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OUTPUT_STREAM_WIDTH_BA" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PUMPED_COMPUTE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SEGMENTLEN" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SIGNED_ACTIVATIONS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SIMD" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WEIGHT_STREAM_WIDTH_BA" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WEIGHT_WIDTH" -parent ${Page_0}


}

proc update_PARAM_VALUE.ACCU_WIDTH { PARAM_VALUE.ACCU_WIDTH } {
	# Procedure called to update ACCU_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ACCU_WIDTH { PARAM_VALUE.ACCU_WIDTH } {
	# Procedure called to validate ACCU_WIDTH
	return true
}

proc update_PARAM_VALUE.ACTIVATION_WIDTH { PARAM_VALUE.ACTIVATION_WIDTH } {
	# Procedure called to update ACTIVATION_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ACTIVATION_WIDTH { PARAM_VALUE.ACTIVATION_WIDTH } {
	# Procedure called to validate ACTIVATION_WIDTH
	return true
}

proc update_PARAM_VALUE.COMPUTE_CORE { PARAM_VALUE.COMPUTE_CORE } {
	# Procedure called to update COMPUTE_CORE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.COMPUTE_CORE { PARAM_VALUE.COMPUTE_CORE } {
	# Procedure called to validate COMPUTE_CORE
	return true
}

proc update_PARAM_VALUE.FORCE_BEHAVIORAL { PARAM_VALUE.FORCE_BEHAVIORAL } {
	# Procedure called to update FORCE_BEHAVIORAL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FORCE_BEHAVIORAL { PARAM_VALUE.FORCE_BEHAVIORAL } {
	# Procedure called to validate FORCE_BEHAVIORAL
	return true
}

proc update_PARAM_VALUE.INPUT_STREAM_WIDTH_BA { PARAM_VALUE.INPUT_STREAM_WIDTH_BA } {
	# Procedure called to update INPUT_STREAM_WIDTH_BA when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.INPUT_STREAM_WIDTH_BA { PARAM_VALUE.INPUT_STREAM_WIDTH_BA } {
	# Procedure called to validate INPUT_STREAM_WIDTH_BA
	return true
}

proc update_PARAM_VALUE.IS_MVU { PARAM_VALUE.IS_MVU } {
	# Procedure called to update IS_MVU when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.IS_MVU { PARAM_VALUE.IS_MVU } {
	# Procedure called to validate IS_MVU
	return true
}

proc update_PARAM_VALUE.MH { PARAM_VALUE.MH } {
	# Procedure called to update MH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MH { PARAM_VALUE.MH } {
	# Procedure called to validate MH
	return true
}

proc update_PARAM_VALUE.MW { PARAM_VALUE.MW } {
	# Procedure called to update MW when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MW { PARAM_VALUE.MW } {
	# Procedure called to validate MW
	return true
}

proc update_PARAM_VALUE.NARROW_WEIGHTS { PARAM_VALUE.NARROW_WEIGHTS } {
	# Procedure called to update NARROW_WEIGHTS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.NARROW_WEIGHTS { PARAM_VALUE.NARROW_WEIGHTS } {
	# Procedure called to validate NARROW_WEIGHTS
	return true
}

proc update_PARAM_VALUE.OUTPUT_STREAM_WIDTH_BA { PARAM_VALUE.OUTPUT_STREAM_WIDTH_BA } {
	# Procedure called to update OUTPUT_STREAM_WIDTH_BA when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OUTPUT_STREAM_WIDTH_BA { PARAM_VALUE.OUTPUT_STREAM_WIDTH_BA } {
	# Procedure called to validate OUTPUT_STREAM_WIDTH_BA
	return true
}

proc update_PARAM_VALUE.PE { PARAM_VALUE.PE } {
	# Procedure called to update PE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PE { PARAM_VALUE.PE } {
	# Procedure called to validate PE
	return true
}

proc update_PARAM_VALUE.PUMPED_COMPUTE { PARAM_VALUE.PUMPED_COMPUTE } {
	# Procedure called to update PUMPED_COMPUTE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PUMPED_COMPUTE { PARAM_VALUE.PUMPED_COMPUTE } {
	# Procedure called to validate PUMPED_COMPUTE
	return true
}

proc update_PARAM_VALUE.SEGMENTLEN { PARAM_VALUE.SEGMENTLEN } {
	# Procedure called to update SEGMENTLEN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SEGMENTLEN { PARAM_VALUE.SEGMENTLEN } {
	# Procedure called to validate SEGMENTLEN
	return true
}

proc update_PARAM_VALUE.SIGNED_ACTIVATIONS { PARAM_VALUE.SIGNED_ACTIVATIONS } {
	# Procedure called to update SIGNED_ACTIVATIONS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SIGNED_ACTIVATIONS { PARAM_VALUE.SIGNED_ACTIVATIONS } {
	# Procedure called to validate SIGNED_ACTIVATIONS
	return true
}

proc update_PARAM_VALUE.SIMD { PARAM_VALUE.SIMD } {
	# Procedure called to update SIMD when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SIMD { PARAM_VALUE.SIMD } {
	# Procedure called to validate SIMD
	return true
}

proc update_PARAM_VALUE.WEIGHT_STREAM_WIDTH_BA { PARAM_VALUE.WEIGHT_STREAM_WIDTH_BA } {
	# Procedure called to update WEIGHT_STREAM_WIDTH_BA when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WEIGHT_STREAM_WIDTH_BA { PARAM_VALUE.WEIGHT_STREAM_WIDTH_BA } {
	# Procedure called to validate WEIGHT_STREAM_WIDTH_BA
	return true
}

proc update_PARAM_VALUE.WEIGHT_WIDTH { PARAM_VALUE.WEIGHT_WIDTH } {
	# Procedure called to update WEIGHT_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WEIGHT_WIDTH { PARAM_VALUE.WEIGHT_WIDTH } {
	# Procedure called to validate WEIGHT_WIDTH
	return true
}


proc update_MODELPARAM_VALUE.IS_MVU { MODELPARAM_VALUE.IS_MVU PARAM_VALUE.IS_MVU } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.IS_MVU}] ${MODELPARAM_VALUE.IS_MVU}
}

proc update_MODELPARAM_VALUE.COMPUTE_CORE { MODELPARAM_VALUE.COMPUTE_CORE PARAM_VALUE.COMPUTE_CORE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.COMPUTE_CORE}] ${MODELPARAM_VALUE.COMPUTE_CORE}
}

proc update_MODELPARAM_VALUE.PUMPED_COMPUTE { MODELPARAM_VALUE.PUMPED_COMPUTE PARAM_VALUE.PUMPED_COMPUTE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PUMPED_COMPUTE}] ${MODELPARAM_VALUE.PUMPED_COMPUTE}
}

proc update_MODELPARAM_VALUE.MW { MODELPARAM_VALUE.MW PARAM_VALUE.MW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MW}] ${MODELPARAM_VALUE.MW}
}

proc update_MODELPARAM_VALUE.MH { MODELPARAM_VALUE.MH PARAM_VALUE.MH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MH}] ${MODELPARAM_VALUE.MH}
}

proc update_MODELPARAM_VALUE.PE { MODELPARAM_VALUE.PE PARAM_VALUE.PE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PE}] ${MODELPARAM_VALUE.PE}
}

proc update_MODELPARAM_VALUE.SIMD { MODELPARAM_VALUE.SIMD PARAM_VALUE.SIMD } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SIMD}] ${MODELPARAM_VALUE.SIMD}
}

proc update_MODELPARAM_VALUE.ACTIVATION_WIDTH { MODELPARAM_VALUE.ACTIVATION_WIDTH PARAM_VALUE.ACTIVATION_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ACTIVATION_WIDTH}] ${MODELPARAM_VALUE.ACTIVATION_WIDTH}
}

proc update_MODELPARAM_VALUE.WEIGHT_WIDTH { MODELPARAM_VALUE.WEIGHT_WIDTH PARAM_VALUE.WEIGHT_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WEIGHT_WIDTH}] ${MODELPARAM_VALUE.WEIGHT_WIDTH}
}

proc update_MODELPARAM_VALUE.ACCU_WIDTH { MODELPARAM_VALUE.ACCU_WIDTH PARAM_VALUE.ACCU_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ACCU_WIDTH}] ${MODELPARAM_VALUE.ACCU_WIDTH}
}

proc update_MODELPARAM_VALUE.NARROW_WEIGHTS { MODELPARAM_VALUE.NARROW_WEIGHTS PARAM_VALUE.NARROW_WEIGHTS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.NARROW_WEIGHTS}] ${MODELPARAM_VALUE.NARROW_WEIGHTS}
}

proc update_MODELPARAM_VALUE.SIGNED_ACTIVATIONS { MODELPARAM_VALUE.SIGNED_ACTIVATIONS PARAM_VALUE.SIGNED_ACTIVATIONS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SIGNED_ACTIVATIONS}] ${MODELPARAM_VALUE.SIGNED_ACTIVATIONS}
}

proc update_MODELPARAM_VALUE.SEGMENTLEN { MODELPARAM_VALUE.SEGMENTLEN PARAM_VALUE.SEGMENTLEN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SEGMENTLEN}] ${MODELPARAM_VALUE.SEGMENTLEN}
}

proc update_MODELPARAM_VALUE.FORCE_BEHAVIORAL { MODELPARAM_VALUE.FORCE_BEHAVIORAL PARAM_VALUE.FORCE_BEHAVIORAL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FORCE_BEHAVIORAL}] ${MODELPARAM_VALUE.FORCE_BEHAVIORAL}
}

proc update_MODELPARAM_VALUE.WEIGHT_STREAM_WIDTH_BA { MODELPARAM_VALUE.WEIGHT_STREAM_WIDTH_BA PARAM_VALUE.WEIGHT_STREAM_WIDTH_BA } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WEIGHT_STREAM_WIDTH_BA}] ${MODELPARAM_VALUE.WEIGHT_STREAM_WIDTH_BA}
}

proc update_MODELPARAM_VALUE.INPUT_STREAM_WIDTH_BA { MODELPARAM_VALUE.INPUT_STREAM_WIDTH_BA PARAM_VALUE.INPUT_STREAM_WIDTH_BA } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.INPUT_STREAM_WIDTH_BA}] ${MODELPARAM_VALUE.INPUT_STREAM_WIDTH_BA}
}

proc update_MODELPARAM_VALUE.OUTPUT_STREAM_WIDTH_BA { MODELPARAM_VALUE.OUTPUT_STREAM_WIDTH_BA PARAM_VALUE.OUTPUT_STREAM_WIDTH_BA } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OUTPUT_STREAM_WIDTH_BA}] ${MODELPARAM_VALUE.OUTPUT_STREAM_WIDTH_BA}
}

