/******************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @brief	Pipelined multi-input adder tree.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *****************************************************************************/

module add_multi import mvu_pkg::*; #(
	int unsigned  N,
	int unsigned  ARG_WIDTH,
	int  ARG_LO = 0,
	int  ARG_HI = 0,

	localparam int unsigned  SUM_WIDTH = sumwidth(N, ARG_WIDTH, ARG_LO, ARG_HI)
)(
	input	logic  clk,
	input	logic  rst,
	input	logic  en,

	input	logic [ARG_WIDTH-1:0]  arg[N],
	output	logic [SUM_WIDTH-1:0]  sum
);

	// Compute the count of decendents for all nodes in the reduction trees.
	typedef int unsigned  leaf_load_t[2*N-1];
	function leaf_load_t init_leaf_loads();
		automatic leaf_load_t  res;
		for(int unsigned  i = 2*N-1; i-- > N-1;)  res[i] = 1;
		for(int unsigned  i =   N-1; i-- >   0;)  res[i] = res[2*i+1] + res[2*i+2];
		return  res;
	endfunction : init_leaf_loads
	localparam leaf_load_t   LEAF_LOAD = init_leaf_loads();

	// Adder Tree
	localparam bit  SIGNED = ARG_LO < 0;
	uwire signed [SUM_WIDTH-1:0]  tree[2*N-1];

	// Assign Leaves
	for(genvar  i = 0; i < N; i++) begin
		assign  tree[N-1 + i] = $signed({ SIGNED && arg[i][ARG_WIDTH-1], arg[i] });
	end

	// Reduction
	//  - Register all fully-occupied tree levels beyond root node
	localparam int unsigned  HI_NODE_REGISTERED = 2**($clog2(N+1)-1)-2;
	for(genvar  i = 0; i < N-1; i++) begin
		localparam int unsigned  LIDX = 2*i+1;
		localparam int unsigned  RIDX = 2*i+2;
		localparam int unsigned  LW = sumwidth(LEAF_LOAD[LIDX], ARG_WIDTH, ARG_LO, ARG_HI);
		localparam int unsigned  RW = sumwidth(LEAF_LOAD[RIDX], ARG_WIDTH, ARG_LO, ARG_HI);
		localparam int unsigned  SW = sumwidth(LEAF_LOAD[   i], ARG_WIDTH, ARG_LO, ARG_HI);
		uwire [LW-1:0]  l = tree[LIDX];
		uwire [RW-1:0]  r = tree[RIDX];
		uwire [SW-1:0]  s;
		if(SIGNED) begin
			uwire signed [SW-1:0]  s0 = $signed(l) + $signed(r);
			assign	s = s0;
		end
		else begin
			assign	s = l + r;
		end

		if((0 < i) && (i <= HI_NODE_REGISTERED)) begin
			logic [SW-1:0]  S = 0;
			always_ff @(posedge clk) begin
				if(rst)      S <= 0;
				else if(en)  S <= s;
			end
			assign  tree[i] = $signed({ SIGNED && S[SW-1], S} );
		end
		else  assign  tree[i] = $signed({ SIGNED && s[SW-1], s} );
	end

	assign	sum = tree[0];

endmodule : add_multi
