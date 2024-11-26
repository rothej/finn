## Imports.
import argparse
from brevitas.nn import QuantConv1d, QuantLinear, QuantIdentity, QuantReLU
from brevitas.core.quant import QuantType
from brevitas.export import export_onnx_qcdq, export_qonnx
import brevitas.onnx as bo
from brevitas.quant_tensor import QuantTensor
import itertools
import matplotlib.pyplot as plt
import numpy as np
import onnx
import os
import pandas as pd
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
import random
from sklearn.preprocessing import LabelEncoder
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset
import warnings
import yaml

# Noise definitions.
awgn_noise = .005
phase_noise = .005

# Load config file - file defines model shapes as well as threshholds and values for training.
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

## Function definitions.

# Layer mapping from config file.
layer_map = {
    'conv1': nn.Conv1d,
    'maxpool1': nn.MaxPool1d,
    'conv2': nn.Conv1d,
    'maxpool2': nn.MaxPool1d,
    'conv3': nn.Conv1d,
    'maxpool3': nn.MaxPool1d,
    'conv4': nn.Conv1d,
    'maxpool4': nn.MaxPool1d,
    'conv5': nn.Conv1d,
    'fc1': nn.Linear,
    'fc2': nn.Linear,
    'fc3': nn.Linear,
    'fc4': nn.Linear,
    'fc5': nn.Linear
}

# Build the model.
def build_model(config, quantization_bits):
    class VGGlike(nn.Module):
        def __init__(self):
            super(VGGlike, self).__init__()
            self.quantization_bits = quantization_bits
            if (quantization_bits == 0): 
                effective_bits = 32
                self.input_quant = QuantIdentity(quant_type=QuantType.FP, bit_width=effective_bits) # FP.
            else:
                self.input_quant = QuantIdentity(quant_type=QuantType.INT, bit_width=quantization_bits) # INT.
            self.layers = nn.Sequential()
            for i, layer_config in enumerate(config['layers']):
                layer_type = list(layer_config.keys())[0]
                layer_params = layer_config[layer_type]
                if layer_type.startswith('conv'):
                    if quantization_bits > 0:
                        layer = QuantConv1d(
                            in_channels=layer_params['in_channels'],
                            out_channels=layer_params['out_channels'],
                            kernel_size=layer_params['kernel_size'],
                            stride=layer_params['stride'],
                            padding=layer_params['padding'],
                            weight_quant_type=QuantType.INT,
                            weight_bit_width=quantization_bits
                        )
                    else:
                        layer = nn.Conv1d(
                            in_channels=layer_params['in_channels'],
                            out_channels=layer_params['out_channels'],
                            kernel_size=layer_params['kernel_size'],
                            stride=layer_params['stride'],
                            padding=layer_params['padding']
                        )
                elif layer_type.startswith('fc'):
                    if layer_params['in_features'] is None:
                        # Calculate the input features based on the previous layer's output.
                        layer_params['in_features'] = self.layers[-1].out_channels * 2
                    if quantization_bits > 0:
                        layer = QuantLinear(
                            in_features=layer_params['in_features'],
                            out_features=layer_params['out_features'],
                            bias=True,
                            weight_quant_type=QuantType.INT,
                            weight_bit_width=quantization_bits
                        )
                    else:
                        layer = nn.Linear(
                            in_features=layer_params['in_features'],
                            out_features=layer_params['out_features']
                        )
                else:
                    layer = layer_map[layer_type](**layer_params)
                self.layers.add_module(f"{layer_type}{i+1}", layer)
            if quantization_bits > 0:
                self.output_quant = QuantIdentity(quant_type=QuantType.INT,
                                                  bit_width=quantization_bits,
                                                  narrow_range=False,
                                                  signed=False)

        def forward(self, x):
            if self.quantization_bits > 0:
                x = self.input_quant(x)
            for layer in self.layers:
                if isinstance(layer, (nn.Conv1d, QuantConv1d)):
                    x = F.relu(layer(x))
                elif isinstance(layer, nn.MaxPool1d):
                    x = layer(x)
                elif isinstance(layer, (nn.Linear, QuantLinear)):
                    if x.dim() > 2:
                        x = x.view(x.size(0), -1)  # Flatten the tensor.
                    x = F.relu(layer(x))
            if self.quantization_bits > 0:
                x = self.output_quant(x)
            return x
    return VGGlike()

# Train the model.
def train_model(inputs, labels, model, data_loader, criterion, optimizer, device, num_epochs, loss_threshold, prune=False):
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        running_loss  = 0.0
        total_batches = 0

        for i, data in enumerate(data_loader, 0):
            inputs, labels = data

            # Move inputs and labels to device.
            inputs = inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # Zero parameter gradients.
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1
        average_loss = running_loss / total_batches
        print(f'      - Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}')
        
        # Stop training when average loss is below the set loss threshold.
        if average_loss < loss_threshold:
            print(f'    - Training stopped - average loss is less than {loss_threshold}')
            break
    
    # Evaluate the model after training. Throw an error if accuracy is not as expected.
    accuracy = evaluate_model(inputs, labels, model, data_loader, device, return_accuracy=True)
    if not prune:
        if accuracy < 59:
            print(f"*** Accuracy after training is less than 99% ({accuracy:.2f}%). Restarting.")
            train_model(inputs, labels, model, train_loader, criterion, optimizer, device, num_epochs, loss_threshold)
    else:
        if accuracy < 59:
            print(f"*** Accuracy after training is less than 99% ({accuracy:.2f}%). Restarting.")
            train_model(inputs, labels, model, train_loader, criterion, optimizer, device, num_epochs, prune_loss_threshold, prune=True)

# Evaluate the model.
def evaluate_model(inputs, labels, model, data_loader, device, return_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    total_time = 0.0
    num_batches_done = 0
    total_inputs = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data

            # Move inputs and labels to device.
            inputs = inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Record start time and end time around model run.
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            outputs = model(inputs)
            end_time.record()

            torch.cuda.synchronize() # Wait for everything to finish running.

            inference_time = start_time.elapsed_time(end_time)
            total_time += inference_time
            num_batches_done += 1

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            total_inputs += inputs.size(0)

    average_time = total_time / num_batches_done    # Average time per batch.
    average_latency = total_time / total_inputs     # Average time per input.
    accuracy = (100 * (correct / total))
    throughput = total_inputs / (total_time / 1e3)  # Convert total time to seconds.

    # Always print these - no debug status checked.
    print(f'  - Eval. Accuracy: {accuracy:.2f}%')
    print(f'  - Eval. Average latency: %.2f milliseconds/input' % average_latency)
    print(f'  - Eval: Throughput: %.2f inputs/second' % throughput)

    if return_accuracy: # For accuracy evaluation if applicable (within training function).
        return accuracy

# Export the model to onnx format.
def export_model(model, input_shape, export_path, batch_size, quantization):
    """
    Export Brevitas model to QONNX format for FINN compatibility and to support low quantization.
    """
    try:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        # CPU copy of the model created for proper device handling.
        model_cpu = model.cpu()

        input_shape = (1, 2, batch_size)  # Shape for the model input (batch_size, channels, sequence_length).
        
        # Create a QuantTensor instance to mark input as bipolar during export.
        input_a = np.random.randint(0, 1, size=input_shape).astype(np.float32)
        input_a = 2 * input_a - 1
        scale = 1.0
        input_t = torch.from_numpy(input_a * scale)

        # for param in model_cpu.parameters():
        #     param.data = param.data.cpu()
        
        # Export to QONNX.
        export_qonnx(
            model_cpu,
            input_t,
            export_path
        )

        # Verify model.
        if os.path.exists(export_path):
            try:
                model = onnx.load(export_path)
                onnx.checker.check_model(model)
                print(f'  - Basic model verification successful')
            except Exception as err:
                print(f'  - Warning: Model verification failed: {str(err)}')
        else:
            print(f'  - Warning: Export file not found')
    
    except Exception as err:
        print(f'*** Error during export: {str(err)}')
        import traceback
        traceback.print_exc()
        raise

    finally:
        qonnx_cleanup(export_path, out_file=export_path)

# Handles signal generation.
class SigGen(object):
    """
    Returns a randomly generated signal of a specific modulation type.
    Must call a class method for the desired modulation.
    """
    def __init__(self, num_symbols, awgn_noise_power, phase_noise_power):
        self.num_symbols = num_symbols
        self.awgn_noise_power = awgn_noise_power
        self.phase_noise_power = phase_noise_power
        
    def generate_noise(self, num_symbols):
        """
        Generates AWGN noise and phase noise.
        """
        awgn_noise = (np.random.randn(num_symbols) + 
                      1j*np.random.randn(num_symbols))/np.sqrt(2)
        phase_noise = np.random.randn(num_symbols) * self.phase_noise_power
        return awgn_noise, phase_noise
    
    def generate_signal(self, symbols, awgn_noise, phase_noise):
        """
        Generates the final signal with noise.
        """
        noisy_signal = symbols + awgn_noise * np.sqrt(self.awgn_noise_power)
        final_signal = noisy_signal * np.exp(1j*phase_noise)
        return final_signal

    def sig_ask(self, num_levels, amplitude_shifts, modulation_label):
        """
        Function for generating data points for an ASK signal.
        """
        x_degree_shift_points = np.random.randint(0, 2, self.num_symbols)
        x_val = np.random.randint(1, num_levels, self.num_symbols)
        x_degrees = x_degree_shift_points * 360/2.0
        x_radians = x_degrees * np.pi/180.0
        x_phase = np.cos(x_radians) + 1j*np.sin(x_radians)
        x_symbols = x_phase * amplitude_shifts

        awgn_noise, phase_noise = self.generate_noise(self.num_symbols)
        final_signal = self.generate_signal(x_symbols, awgn_noise, phase_noise)

        train_data = [[sig.real, sig.imag, modulation_label] for sig in final_signal]
        return train_data, final_signal

    def sig_4ask(self):
        return self.sig_ask(3, ((np.random.randint(1, 3, self.num_symbols)*0.666667)-0.333333), '4ask')

    def sig_8ask(self):
        return self.sig_ask(5, ((np.random.randint(1, 5, self.num_symbols)*0.285714)-0.142857), '8ask')

    def generate_psks(self, num_shifts, modulation_label):
        """
        Function for generating data points for a PSK signal.
        """
        x_degree_shift_points = np.random.randint(0, num_shifts, self.num_symbols)
        x_degrees = x_degree_shift_points * 360/num_shifts

        x_radians = x_degrees * np.pi/180.0
        x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians)

        awgn_noise, phase_noise = self.generate_noise(self.num_symbols)
        final_signal = self.generate_signal(x_symbols, awgn_noise, phase_noise)

        train_data = [[sig.real, sig.imag, modulation_label] for sig in final_signal]

        return train_data, final_signal

    def generate_qam(self, num_shifts, modulation_label):
        """
        Function for generating data points for a circular QAM signal.
        """
        x_degree_shift_points = np.random.randint(0, num_shifts, self.num_symbols)
        x_val = np.ones(self.num_symbols)
        x_shift = np.zeros(self.num_symbols)

        # For each degree shift completed, amplitude should be
        # appropriate for a QAM signal.
        for n in range(0, self.num_symbols):
            if x_degree_shift_points[n] % 2 == 1:
                x_shift[n] = x_val[n]/2.0
            else:
                x_shift[n] = x_val[n]/1.0

        x_degrees = x_degree_shift_points * 360/num_shifts
        x_radians = x_degrees * np.pi/180.0
        x_phase = np.cos(x_radians) + 1j*np.sin(x_radians)
        x_symbols = x_phase * x_shift

        awgn_noise, phase_noise = self.generate_noise(self.num_symbols)
        final_signal = self.generate_signal(x_symbols, awgn_noise, phase_noise)

        train_data = [[sig.real, sig.imag, modulation_label] for sig in final_signal]

        return train_data, final_signal
    
    def generate_16qam(self):
        """
        Function for generating data points for a 16-QAM signal.
        """
        i_levels = np.array([-3, -1, 1, 3])
        q_levels = np.array([-3, -1, 1, 3])

        i_points = np.random.choice(i_levels, self.num_symbols)
        q_points = np.random.choice(q_levels, self.num_symbols)

        symbols = i_points/3.0 + 1j * q_points/3.0

        awgn_noise, phase_noise = self.generate_noise(self.num_symbols)
        final_signal = self.generate_signal(symbols, awgn_noise, phase_noise)

        train_data = [[sig.real, sig.imag, '16qam'] for sig in final_signal]

        return train_data, final_signal

    def generate_32qam(self):
        """
        Function to generate data points for a 32-QAM signal.
        """
        # Define the I and Q component levels.
        i_levels = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
        q_levels = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])

        # Define the points to be excluded
        invalid_points = {(-3.5, 3.5), (-3.5, 1.5), (-3.5, -0.5), (-3.5, -2.5),
                          (-2.5, 2.5), (-2.5, 0.5), (-2.5, -1.5), (-2.5, -3.5),
                          (-1.5, 3.5), (-1.5, 1.5), (-1.5, -0.5), (-1.5, -2.5),
                          (-0.5, 2.5), (-0.5, 0.5), (-0.5, -1.5), (-0.5, -3.5),
                          (0.5, 3.5),  (0.5, 1.5),  (0.5, -0.5),  (0.5, -2.5),
                          (1.5, 2.5),  (1.5, 0.5),  (1.5, -1.5),  (1.5, -3.5),
                          (2.5, 3.5),  (2.5, 1.5),  (2.5, -0.5),  (2.5, -2.5),
                          (3.5, 2.5),  (3.5, 0.5),  (3.5, -1.5),  (3.5, -3.5)}

        # Get only the valid symbols.
        valid_symbols = [s for s in itertools.product(i_levels, q_levels) if s not in invalid_points]

        # Calculate average power (for normalization).
        avg_power = np.mean([abs(x[0] + 1j*x[1])**2 for x in valid_symbols])

        # Generate random indices and select corresponding symbols.
        symbols_indices = np.random.choice(len(valid_symbols), self.num_symbols)
        symbols = np.array(valid_symbols)[symbols_indices]

        # Convert symbols to complex numbers and normalize.S
        complex_symbols = (symbols[:,0] + 1j * symbols[:,1]) / np.sqrt(avg_power)

        awgn_noise, phase_noise = self.generate_noise(self.num_symbols)
        final_signal = self.generate_signal(complex_symbols, awgn_noise, phase_noise)

        train_data = [[sig.real, sig.imag, '32qam'] for sig in final_signal]

        return train_data, final_signal

    def sig_bpsk(self):
        """
        Function to generate data points for a BPSK signal.
        """
        return self.generate_psks(2, 'bpsk')

    def sig_qpsk(self):
        """
        Function to generate data points for a QPSK signal.
        """
        return self.generate_psks(4, 'qpsk')

    def sig_8psk(self):
        """
        Function to generate data points for a 8PSK signal.
        """
        return self.generate_psks(8, '8psk')
    
    def sig_16psk(self):
        """
        Function to generate data points for a 16PSK signal.
        """
        return self.generate_psks(16, '16psk')

    def sig_8qam(self):
        """
        Function to generate data points for a 8QAM signal.
        """
        return self.generate_qam(8, '8qam')
    
    def sig_16qam(self):
        """
        Routes to the 16QAM generation function.
        """
        return self.generate_16qam()
    
    def sig_32qam(self):
        """
        Routes to the 32QAM generation function.
        """
        return self.generate_32qam()
    
# Splits dataset into chunks.
def split_into_chunks(dataset, chunk_size = 32):
    chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]

    # Grab the label from the first point in each chunk.
    labels = [chunk[0][2] for chunk in chunks]

    # Remove the labels from each datapoint in the chunk.
    chunks = [[[datapoint[0], datapoint[1]] for datapoint in chunk] for chunk in chunks]

    return list(zip(chunks, labels))

# Plots the generated training datasets.
def plot_datasets():
    datasets_plot = {
        "4ASK Signal" : siggen_inst_train.sig_4ask,
        "8ASK Signal" : siggen_inst_train.sig_8ask,
        "BPSK Signal" : siggen_inst_train.sig_bpsk,
        "QPSK Signal" : siggen_inst_train.sig_qpsk,
        "8PSK Signal" : siggen_inst_train.sig_8psk,
        "16PSK Signal": siggen_inst_train.sig_16psk,
        "8QAM Signal" : siggen_inst_train.sig_8qam,
        "16QAM Signal": siggen_inst_train.sig_16qam,
        "32QAM Signal": siggen_inst_train.sig_32qam
    }

    # Generate the datasets.
    datasets_plot = {title: func()[0] for title, func in datasets_plot.items()}

    # Define the number of rows and columns needed for subplots.
    num_rows = int(np.ceil(len(datasets_plot) / 3))

    # Initialize the subplot.
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))
    fig.suptitle("Modulated Training Signal Data Plots. X = I, Y = Q")
    fig.tight_layout(pad=3.0)

    # Loop over the datasets.
    for i, (title, datasets_plot) in enumerate(datasets_plot.items()):
        # Calculate the subplot position.
        row = i // num_rows
        col = i % 3
        # Plot the dataset (with smaller data points).
        axs[row, col].plot([x[0] for x in datasets_plot], [x[1] for x in datasets_plot], '.', markersize=2)
        axs[row, col].set_title(title)
        axs[row, col].set_xlim(-1.5,1.5)
        axs[row, col].set_ylim(-1.5,1.5)
        axs[row, col].grid(True)

    # Show plot.
    plt.show()

# Encodes labels and converts to dataset for Dataloader to use.
def encode_labels(chunks_df, batch_size):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(chunks_df['Labels'])

    # Replace labels in the DataFrame with encoded labels.
    chunks_df['Labels'] = encoded_labels

    # Get the unique class labels.
    unique_labels = label_encoder.classes_

    # Create a dictionary to map encoded values to label strings.
    encoded_to_label = {encoded: label for encoded, label in enumerate(unique_labels)}

    # Print the encoded values of different labels.
    if (debug):
        for label in unique_labels:
            encoded_value = label_encoder.transform([label])[0]
            print(f"Encoded value for '{label}': {encoded_value}")
    
    chunks_as_list   = chunks_df['Chunks'].tolist()
    chunks_as_tensor = torch.tensor(chunks_as_list)
    labels_as_tensor = torch.tensor(chunks_df['Labels'].values, dtype=torch.long)

    # Print sequence length.
    if (debug):
        print(chunks_as_tensor.shape)
    
    # Create a TensorDataset from the DataFrame.
    chunks_as_tensor_permuted = chunks_as_tensor.permute(0, 2, 1).to(device)
    dataset = TensorDataset(chunks_as_tensor_permuted, labels_as_tensor.to(device))

    # Create a DataLoader for batching.
    data_loader = DataLoader(dataset, batch_size, shuffle=True)

    # Verify that it is instantiated.
    if (debug):
        print(data_loader)
    
    return data_loader

# Prunes the less useful weights of the model, if applicable.
def prune_model(model, amount): # Amount is percent. So 0.1 is 10%, etc.
    # This needs to be dynamic based on model type.
    layers_to_prune = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5, model.fc1, model.fc2, model.fc3, model.fc4, model.fc5]
    for layer in layers_to_prune:
        # Prune the layer based on L1 norm.
        prune.ln_structured(layer, name="weight", amount=amount, n=1, dim=0)
        # Make the pruning permanent.
        prune.remove(layer, 'weight')

## Main task flow.

# Enable CUDA if possible.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(f'- CUDA enabled status: ', use_cuda)

# Parse command-line arguments.
parser = argparse.ArgumentParser(description='Train and benchmark models.')
parser.add_argument('--model', type=str, default=None, help='Name of the specific model to run (e.g. vgglike_5f_5c_4re_4mp) (default: run all models)')
parser.add_argument('--quant', type=int, default=None, help='Quantization level (0, 8, 6, 4) (default: run all quantization levels)')
parser.add_argument('--prune', type=float, default=None, help='Pruning percentage (0, 0.2, 0.3) (default: run all pruning levels)')
parser.add_argument('--debug', type=str, default=0, help='Set to 1 to enable debug statements. (default: 0)')
args = parser.parse_args()

debug = args.debug
if (debug == 0):
    # Disable warnings (looking at you, PyTorch).
    warnings.filterwarnings("ignore")
inputs = 0
labels = 0
batch_size = 32

# Iterate over the model configurations.
for model_config in config['models']:
    model_name = model_config['name']

    # Skip the model if it doesn't match the specified model name (if applicable).
    if args.model is not None and model_name != args.model:
        continue

    print(f'  - Training model: {model_name}')

    # Build the model, and apply quantization if needed.
    for pruning_config in model_config['pruning']:
        pruning = pruning_config['percentage']

        # Skip the pruning level if it doesn't match the specified pruning percentage (if applicable).
        if args.prune is not None and pruning != args.prune:
            continue

        for quant_config in model_config['quantization']:
            quantization = quant_config['bits']

            # Skip the quantization level if it doesn't match the specified quantization bits (if applicable).
            if args.quant is not None and quantization != args.quant:
                continue

            print(f'    - Pruning: {pruning}')
            print(f'    - Quantization: {quantization}')

            while True:
                try:
                    model = build_model(model_config, quantization)
                    model.to(device)
                    # if (debug): print(model)
                    if (debug): print(device)

                    # Instantiate signal generator and generate datasets for training and evaluation.
                    siggen_inst_train = SigGen(16384, awgn_noise, phase_noise)
                    siggen_inst_eval  = SigGen(16384, awgn_noise, phase_noise)

                    datasets_funcs_train = [siggen_inst_train.sig_4ask, siggen_inst_train.sig_8ask,
                                            siggen_inst_train.sig_bpsk, siggen_inst_train.sig_qpsk,
                                            siggen_inst_train.sig_8psk, siggen_inst_train.sig_16psk,
                                            siggen_inst_train.sig_8qam, siggen_inst_train.sig_16qam, 
                                            siggen_inst_train.sig_32qam]
                    datasets_funcs_eval  = [siggen_inst_eval.sig_4ask, siggen_inst_eval.sig_8ask,
                                            siggen_inst_eval.sig_bpsk, siggen_inst_eval.sig_qpsk,
                                            siggen_inst_eval.sig_8psk, siggen_inst_eval.sig_16psk,
                                            siggen_inst_eval.sig_8qam, siggen_inst_eval.sig_16qam, 
                                            siggen_inst_eval.sig_32qam]

                    # Shuffle datasets and split into chunks.
                    dataset_chunks_labels_train = []
                    for func in datasets_funcs_train:
                        dataset_train, _ = func()
                        random.shuffle(dataset_train) 
                        dataset_chunks_labels_train += split_into_chunks(dataset_train)
                    dataset_chunks_labels_eval = []
                    for func in datasets_funcs_eval:
                        dataset_eval, _ = func()
                        random.shuffle(dataset_eval) 
                        dataset_chunks_labels_eval += split_into_chunks(dataset_eval)

                    # Shuffle chunks with labels attached.
                    random.shuffle(dataset_chunks_labels_train)
                    random.shuffle(dataset_chunks_labels_eval)

                    # Separate chunks and labels.
                    all_chunks_train, all_labels_train = zip(*dataset_chunks_labels_train)
                    all_chunks_eval, all_labels_eval   = zip(*dataset_chunks_labels_eval)

                    # Converts chunks and labels into dataframe format datasets into one.
                    chunks_df_train = pd.DataFrame({'Chunks': all_chunks_train, 'Labels': all_labels_train})
                    chunks_df_eval  = pd.DataFrame({'Chunks': all_chunks_eval, 'Labels': all_labels_eval})

                    if (debug):
                        print(chunks_df_train)
                        plot_datasets()

                    # Data loaders.
                    train_loader = encode_labels(chunks_df_train, batch_size)
                    eval_loader  = encode_labels(chunks_df_eval, batch_size)

                    # Define loss function and optimizer.
                    criterion = nn.CrossEntropyLoss()           # Cross entropy loss function.
                    lr = quant_config['learning_rate']          # Initial learning rate for Adam Optimizer.
                    weight_decay = quant_config['weight_decay'] # For L2 regularization, to prevent overfitting.
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    if (debug): print(optimizer)

                    # Train the model.
                    num_epochs     = quant_config['num_epochs']
                    loss_threshold = quant_config['loss_threshold']
                    train_model(inputs, labels, model, train_loader, criterion, optimizer, device, num_epochs, loss_threshold)

                    # Prune and retrain again, if applicable.
                    if pruning != 0:
                        prune_model(model, pruning)
                        prune_loss_threshold = pruning_config['prune_loss_threshold']
                        train_model(inputs, labels, model, train_loader, criterion, optimizer, device, num_epochs, prune_loss_threshold, prune=True)

                    # Export the model to ONNX format.
                    input_shape = (1, 2, 32)
                    if quantization == 0:
                        export_path = f'models/{model_name}_pr{pruning}_noquant.onnx'
                    else:
                        export_path = f'models/{model_name}_pr{pruning}_quant{quantization}.onnx'
                    export_model(model, input_shape, export_path, batch_size, quantization)

                    break

                except ValueError as err:
                    print(f"*** Error: {str(err)}")
                    # Prompt for new configuration values.
                    lr             = float(input("--> Enter a new learning rate (current: {}): ".format(model_config['learning_rate'])))
                    weight_decay   = float(input("--> Enter a new weight decay (current: {}): ".format(model_config['weight_decay'])))
                    loss_threshold = float(input("--> Enter a new loss threshold (current: {}): ".format(model_config['loss_threshold'])))
                    num_epochs     = int(input("--> Enter a new number of epochs (current: {}): ".format(model_config['num_epochs'])))

                    # Update the configuration with new values.
                    model_config['learning_rate']  = lr
                    model_config['weight_decay']   = weight_decay
                    model_config['loss_threshold'] = loss_threshold
                    model_config['num_epochs']     = num_epochs

                    if pruning != 0:
                        prune_loss_threshold = float(input("--> Enter a new prune loss threshold (current: {}): ".format(model_config['prune_loss_threshold'])))
                        model_config['prune_loss_threshold'] = prune_loss_threshold

print('- Training and benchmarking completed.')