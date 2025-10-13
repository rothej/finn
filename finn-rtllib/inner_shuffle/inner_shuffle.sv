/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief       A streaming 2D parallel transpose unit. (I,J) -> (J,I) with SIMD
 * 		parallelism
 * @author      Shane T. Fleming <shane.fleming@amd.com>
 *
 * @description
 *
 * This unit can perform a streaming transpose (I,J) -> (J,I) with SIMD
 * parallelism.
 * It achieves this by using SIMD banks of memory and rotating write and reads
 * to the banks such that collisions are avoided and maximum throughput can be
 * maintained (II=1).
 *
 * Decisions about when to rotate writes and reads to the different banks are
 * made by a WR_ROT_PERIOD param, for writes, and a RD_PATTERN param matrix, for reads.
 * These two are computed at elaboration time and are constants at runtime.
 *
 * After WR_ROT_PERIOD writes to the banks the write bank allocation is shifted to
 * the right by one position.
 * The WR_ROT_PERIOD is determined by considering the GCD of SIMD
 * along with the inner input dimension J.
 *
 * The RD_PATTERN for the read side is a SIMDxSIMD matrix of banks that is a
 * periodic pattern of banks across the input matrix. This is computed by
 * evaluating what a SIMDxSIMD block of bank allocations will look like with
 * the current WR_ROT_PERIOD.
 *
 * On the write path of the hardware data is written into the banks according
 * to the initial write banks. A counter tracks how many writes have happened
 * and then after WR_ROT_PERIOD counts the banks are rotated. The write
 * address is incremented by one every write for every bank.
 *
 * The Read path has logic to generate the addresses for SIMD reads based on
 * the current index of the output loop:
 *
 *        	j : [0,J)
 *        	   i : [0,I)
 *        	     emit(i*J + j)
 *
 * SIMD addresses are generated and each is sent to the appropriate SIMD banks
 * based on the schedule in the relevant column of the RD_PATTERN matrix.
 * This column of the RD_PATTERN matrix is then forwarded to the output of the
 * banks, where a clock cycle later the relevant outputs appear at each bank
 * output. The output data is then rearranged again using the forwarded RD_PATTERN
 * column to assign the appropriate output signals.
 * Logic is used to track what column of the the RD_PATTERN to use based
 * on where the circuit current is in the output iteration space.
 *
 * Control flow for writing and reading the banks are managed by job
 * scheduling logic. This means that while a job is being
 * outputted on the read side, the next job can be written on the write side
 * enabling both the write path and the read path to be active simultaneously.
****************************************************************************/

// A memory bank in the inner_shuffle design. Pattern was kept as simple
// as possible to help with Vivado BRAM inference.
module mem_bank #(
	int unsigned WIDTH,
	int unsigned DEPTH
)(
	input logic clk,
	input logic rst,

	input logic [WIDTH-1:0] d_in,
	input logic [$clog2(DEPTH)-1:0] wr_addr,
	input logic wr_en,

	output logic [WIDTH-1:0] d_out,
	input  logic [$clog2(DEPTH)-1:0] rd_addr,
	input  logic rd_hold
);

	(* ram_style="block" *) logic [WIDTH-1:0] Mem [DEPTH-1:0]; // The Mem for this bank

	// Write channel
	always_ff @(posedge clk)
		if (wr_en) Mem[wr_addr] <= d_in;

	// Read channel
	always_ff @(posedge clk)
		if (rst)
			d_out <= 'd0;
		else
			if(!rd_hold)
				d_out <= Mem[rd_addr];
endmodule


// ----------------------------------------
// Parallel Transpose Unit (InnerShuffle)
// ----------------------------------------
module inner_shuffle #(
	int unsigned BITS,   // Bitwidth of each element
	int unsigned I   ,   // Input dimension I
	int unsigned J   ,   // Input dimension J
	int unsigned SIMD    // SIMD parallelism
)(
	input logic                       clk, // global control
	input logic                       rst,

	output logic                      irdy, // Input stream
	input  logic                      ivld,
	input  logic [SIMD-1:0][BITS-1:0] idat,

	input  logic                      ordy, // Output stream
	output logic                      ovld,
	output logic [SIMD-1:0][BITS-1:0] odat
);


	// assertion checks for ensuring that the constraints are satisfied
	initial begin
		if (I%SIMD != 0) begin
			$fatal(1, "Error! Assertion I%SIMD == 0 is not met for this circuit");
		end
	end

	function int unsigned gcd(input int a, input int b);
		return (b == 0) ? a : gcd(b, a%b);
	endfunction

	// elaboration time compute for generating the WR_ROT_PERIOD
	// This is used to determine how often the write banks should be
	// rotated at runtime, i.e. after how many SIMD writes into the banks
	// do we need to swap the allocation.
	function automatic logic [$clog2(I*J)-1: 0] calculate_WR_ROT_PERIOD();
		if (gcd(J,SIMD) > 1)
			return J / gcd(J,SIMD);
		else
			return 0;
	endfunction : calculate_WR_ROT_PERIOD

	localparam logic [$clog2(I*J)-1: 0] WR_ROT_PERIOD = calculate_WR_ROT_PERIOD();
	localparam logic [$clog2(I*J)-1: 0] RD_ROT_PERIOD = I/SIMD; // (I % SIMD == 0) is a constraint
	typedef logic [$clog2(SIMD)-1:0] rd_pattern_col_t [SIMD-1:0];

	// --------------------------------------------------------------------------
	// RD_INITIAL_PATTERN & RD_PERMUTATION_PATTERN
	// --------------------------------------------------------------------------
	function automatic rd_pattern_col_t generate_initial_rd_pattern();
		rd_pattern_col_t rd_pat_0; // The RD Pattern for the first column
		for(int unsigned i=0; i<SIMD; i++) begin
			if(WR_ROT_PERIOD != 0)
				rd_pat_0[i] = ( (i*J)%SIMD + (i*J)/(WR_ROT_PERIOD*SIMD) ) % SIMD;
			else
				rd_pat_0[i] = (i*J)%SIMD;
		end
		return rd_pat_0;
	endfunction : generate_initial_rd_pattern


	typedef logic [$clog2(SIMD)-1:0] rd_perm_t [SIMD-1:0];
	function automatic rd_perm_t generate_rd_permutation_pattern();
		rd_perm_t perm_pattern;

		rd_pattern_col_t rd_pat_0 = generate_initial_rd_pattern();
		rd_pattern_col_t rd_pat_1; // The RD Pattern for the second column
		for(int unsigned i=0; i<SIMD; i++) begin
			rd_pat_1[i] = ( rd_pat_0[i] + 1 )%SIMD;
		end

		// Calculate permutation indices
		foreach (rd_pat_0[i])
			foreach (rd_pat_1[j])
				if (rd_pat_0[i] == rd_pat_1[j]) begin
					perm_pattern[i] = j;
					break;
				end
		return perm_pattern;
	endfunction : generate_rd_permutation_pattern

	localparam rd_pattern_col_t rd_init_pat = generate_initial_rd_pattern();
	localparam rd_perm_t rd_perm_pat        = generate_rd_permutation_pattern();
	rd_pattern_col_t rd_pat;

	// --------------------------------------------------------------------------
	//   Memory Banks
	// --------------------------------------------------------------------------
	logic OsbVld; // output skidbuffer valid signal
	logic OsbVldD; // output skidbuffer valid signal
	logic OsbRdy; // output skid buffer ready signal

	localparam int unsigned BANK_DEPTH = 2*(I*J/SIMD);
	localparam int unsigned PAGE_OFFSET =   (I*J)/SIMD;

	// Instantiate separate banks
	logic                           mem_banks_wr_en   [SIMD-1:0];
	logic [BITS-1:0]                mem_banks_in      [SIMD-1:0];
	logic [BITS-1:0]                mem_banks_out     [SIMD-1:0];
	logic [$clog2(BANK_DEPTH)-1:0]  mem_banks_rd_addr [SIMD-1:0];
	logic [$clog2(BANK_DEPTH)-1:0]  wr_addr;

	// Generates the SIMD dual port memory banks
	for(genvar i =0; i<SIMD; i++) begin : gen_mem_banks
		mem_bank #(
			.WIDTH(BITS),
			.DEPTH(BANK_DEPTH)
		) mem_bank_inst (
			.clk(clk),
			.rst(rst),
			.d_in(mem_banks_in[i]),
			.wr_addr(wr_addr),
			.wr_en(irdy && ivld),
			.d_out(mem_banks_out[i]),
			.rd_addr(mem_banks_rd_addr[i]),
			.rd_hold(!OsbRdy)
		);
	end : gen_mem_banks

	// Write bank schedule
	logic[$clog2(SIMD)-1:0] wr_bank_schedule      [SIMD-1:0];
	logic[$clog2(SIMD)-1:0] next_wr_bank_schedule [SIMD-1:0];

	// Rotate the next write schedule (only registered every WR_ROT_PERIOD)
	// This is reset every SIMD rows written
	always_comb begin : writeBankScheduleRotation
		// Reset the write bank allocation after SIMD Rows
		if (WrBankReset == J-1)
			for(int unsigned i=0; i<SIMD; i++) next_wr_bank_schedule[i] = i;
		else begin
			next_wr_bank_schedule [SIMD-1] = wr_bank_schedule[0];
			for(int unsigned i=0; i<SIMD-1; i++)
				next_wr_bank_schedule[i] = wr_bank_schedule[i+1];
		end

	end : writeBankScheduleRotation

	// Remap the input based on the current write bank rotation
	always_comb begin
		for(int unsigned i=0; i<SIMD; i++)  mem_banks_in[i] = 'd0;  // default values to avoid latch inference
		for(int unsigned i=0; i<SIMD; i++)  mem_banks_in[wr_bank_schedule[i]] = idat[i];
	end

	// Write bank schedule rotation logic
	logic[$clog2(WR_ROT_PERIOD)-1:0] WrRotCounter;
	logic[$clog2(I*J/SIMD)-1:0]      WrCounter;

	// Bank schedule reset (Resets the bank write after SIMD*I elements written)
	logic[$clog2(J)-1:0]      	 WrBankReset;

	always_ff @(posedge clk) begin
		if (rst) begin
			for(int unsigned i=0; i<SIMD; i++) wr_bank_schedule[i] <= i;
			WrRotCounter <= 'd0;
			WrCounter <= 'd0;
			WrBankReset <= 'd0;
		end
		else
			if (ivld && irdy) begin // Detect once we need to rotate and perform right rotation

				if(WrBankReset == J-1)
					WrBankReset <= 'd0;
				else
					WrBankReset <= WrBankReset + 'd1;

				if (WrRotCounter == WR_ROT_PERIOD - 1) begin
					WrRotCounter <= 'd0;
					if (WrCounter == (I*J/SIMD - 1))
						WrCounter <= 'd0;
					for (int unsigned i = 0; i < SIMD; i++) wr_bank_schedule[i] <= next_wr_bank_schedule[i];
				end
				else begin
					WrRotCounter <= WrRotCounter + 'd1;
					WrCounter <= WrCounter + 'd1;
				end
			end
	end

	// Job tracking and bank page locking
	logic [1:0] WrJobsDone; // Bit vector tracking when writes have been completed to pages
	logic RdPageInProgress; // 0 - reading from PAGE A, 1 - reading from PAGE B
	logic [$clog2(BANK_DEPTH)-1:0] PageRdOffset;

	always_ff @(posedge clk) begin
		if (rst) begin
			WrJobsDone <= 2'b00;
			RdPageInProgress <= 1'b0;
		end

		// Track if we have completed a job
		if (wr_addr == PAGE_OFFSET   - 1) WrJobsDone[0] <= 1'b1;
		if (wr_addr == 2*PAGE_OFFSET - 1) WrJobsDone[1] <= 1'b1;

		// Clear the relevant job once it is read
		if ((RdJCnt == J-1) && (RdICnt+SIMD == I) && (OsbRdy && OsbVldD)) begin
		       WrJobsDone[RdPageInProgress] <= 1'b0;
		       RdPageInProgress <= !RdPageInProgress;
		end
	end

	assign PageRdOffset = RdPageInProgress ? PAGE_OFFSET : 'd0;
	assign irdy = !WrJobsDone[0] || !WrJobsDone[1];

	// Write address incrementer (resets to the start once the second page is written)
	always_ff @(posedge clk) begin
		if (rst) wr_addr <= 'd0;
		else
			if (ivld && irdy)
				if (wr_addr < (2*PAGE_OFFSET - 1))
					wr_addr <= wr_addr + 'd1;
				else
					wr_addr <= 'd0;
	end

	// --------------------------------------------------------------------------
	//    Read Address generation
	// --------------------------------------------------------------------------
	logic[$clog2(I)-1 : 0] RdICnt;
	logic[$clog2(J)-1 : 0] RdJCnt;
	logic RdGuard;
	assign RdGuard = !RdPageInProgress && !WrJobsDone[0] && !WrJobsDone[1];

	// Logic to track which iteration we are on for the read side
	always_ff @(posedge clk) begin : readIndexLoopTracking
		if (rst) begin
			RdICnt <= 'd0;
			RdJCnt <= 'd0;
		end
		else
			if(OsbRdy && !RdGuard)
				if((RdICnt+SIMD) >= I) begin
					RdICnt <= 'd0;
					if( RdJCnt < J-1)
						RdJCnt <= RdJCnt + 'd1;
					else
						RdJCnt <= 'd0;
				end
				else
					RdICnt <= RdICnt + SIMD;
	end : readIndexLoopTracking

	// Combinatorial generation of the current set of Read addresses
	always_comb begin : bankRdAddrGen
		for(int unsigned i=0; i<SIMD; i++) mem_banks_rd_addr[i] = 'd0; // default to avoid latch inference
		for(int unsigned i=0; i < SIMD; i++)
			mem_banks_rd_addr[rd_pat[i]] = ((RdICnt + i)*J + RdJCnt)/SIMD + PageRdOffset;
	end : bankRdAddrGen
	// --------------------------------------------------------------------------

	// --------------------------------------------------------------------------
        logic [SIMD-1:0][BITS-1:0]     data_reg; // remapped output
	logic [$clog2(I*J/SIMD)-1:0]   rd_pattern_idx;
	rd_pattern_col_t               rd_pattern_col_ff; // The fowarded rotation pattern

	// Forward the current RD_PATTERN row onto the next pipeline stage
	always_ff @(posedge clk) begin : rdPatternColForwarding
		if (rst) OsbVld <= 0;
		else begin
			OsbVld <= !RdGuard;
			OsbVldD <= OsbVld;
			if (OsbRdy && !RdGuard)
				for(int unsigned i=0; i<SIMD; i++)
					rd_pattern_col_ff[i] <= rd_pat[i];
		end
	end : rdPatternColForwarding

	// Structural remapping using the output of the memory banks
	// and the Read rotation from the previous clock cycle that was
	// used to generate the read addresses.
	for(genvar i=0; i<SIMD; i++)
		assign data_reg[i] = mem_banks_out[rd_pattern_col_ff[i]];
	// --------------------------------------------------------------------------

	// --------------------------------------------------------------------------
	logic [$clog2(I*J/SIMD)-1:0] rd_counter;

	// the next permutation of the rd pattern
	rd_pattern_col_t rd_pat_next;
	always_comb begin
		for(int unsigned i=0; i<SIMD; i++) rd_pat_next[i] = 'd0; // default to avoid latch inference
		for(int unsigned i=0; i<SIMD; i++) rd_pat_next[rd_perm_pat[i]] = rd_pat[i];
	end

	// Track the read count for determining when rotations should occur.
	always_ff @(posedge clk) begin : readTrackingForRotationDecisions
		if (rst) begin
			rd_counter <= 'd0;
			for(int unsigned i=0; i<SIMD; i++)
				rd_pat[i] <= rd_init_pat[i];
		end
		else begin
			if (OsbRdy && !RdGuard) begin
				rd_counter <= rd_counter + 'd1;
				if (rd_counter == RD_ROT_PERIOD-1) begin
					rd_counter <= 'd0;
					for(int unsigned i=0; i<SIMD; i++)
						rd_pat[i] <= rd_pat_next[i];
				end

				// At the page boundary reset our RD_PATTERN lookup
				if ((RdJCnt == J-1) && (RdICnt+SIMD == I )) begin
					rd_counter <= 'd0;
					for(int unsigned i=0; i<SIMD; i++)
						rd_pat[i] <= rd_init_pat[i];
				end
			end

		end
	end : readTrackingForRotationDecisions

	// --------------------------------------------------------------------------

	// Output SkidBuffer -- Used to decouple control signals for timing
	// improvements
	skid #(
		.DATA_WIDTH(SIMD*BITS)
	)
	oskidbf_inst (
		.clk(clk),
		.rst(rst),

		.idat(data_reg),
		.ivld(OsbVld),
		.irdy(OsbRdy),

		.odat(odat),
		.ovld(ovld),
		.ordy(ordy)
	);

endmodule : inner_shuffle
