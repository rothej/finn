from finn.builder.build_dataflow import build_dataflow_cfg
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.util.cleanup import cleanup_model

model = ModelWrapper("models/vgglike_5f_5c_4re_4mp_pr0_noquant.onnx")
model = cleanup_model(model)

model = model.transform(ConvertQONNXtoFINN())



# Configure the build
build_config = {
    # Target FPGA board
    "board": "Pynq-Z1",  # or your specific target board
    
    # Performance targets
    "target_fps": 100000,  # desired frames per second
    "synth_clk_period_ns": 10.0,  # clock period in nanoseconds
    
    # Build steps to include
    "generate_outputs": [
        "estimate_reports",     # Get resource estimates
        "stitched_ip",         # Generate the IP core
        "hw_emu_bitfile",      # Hardware emulation
        "bitfile",             # Final FPGA bitstream
    ],
    
    # Output directory
    "output_dir": "build_results",
    
    # Optional: FPGA synthesis settings
    "shell_flow_type": "vivado",
    "vitis_platform": "platform_name"  # if using Vitis
}

# Run the build
build_dataflow_cfg(model, build_config)

# Verify functionality before synthesis.
from finn.core.onnx_exec import execute_onnx

input_tensor = np.random.rand(1, 2, 32).astype(np.float32)
output_dict = execute_onnx(model, {"input": input_tensor})

# Generate performance estimates.
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
res_est = hls_synth_res_estimation(model)

# Analyze the model.
from finn.analysis.topology import topology_list
topology = topology_list(model)