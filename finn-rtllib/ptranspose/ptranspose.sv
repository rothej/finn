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

// A memory bank in the ptranspose design. Pattern was kept as simple
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

	(* ram_style="block" *) logic [WIDTH-1:0] mem [DEPTH-1:0]; // The Mem for this bank

	// Write channel
	always_ff @(posedge clk) 
		if (wr_en) mem[wr_addr] <= d_in;
	
	// Read channel
	always_ff @(posedge clk) 
		if (rst) 
			d_out <= 'd0;
		else 
			if(!rd_hold) 
				d_out <= mem[rd_addr];
endmodule


// ----------------------------------------
// Parallel Transpose Unit (PTranspose)
// ----------------------------------------
module ptranspose #(
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
	logic osb_vld; // output skidbuffer valid signal
	logic osb_vld_d; // output skidbuffer valid signal
	logic osb_rdy; // output skid buffer ready signal

	localparam int unsigned BANK_DEPTH  = 2*(I*J/SIMD);
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
			.rd_hold(!osb_rdy)
		);
	end : gen_mem_banks

	// Write bank schedule	
	logic[$clog2(SIMD)-1:0] wr_bank_schedule      [SIMD-1:0];
	logic[$clog2(SIMD)-1:0] next_wr_bank_schedule [SIMD-1:0];

	// Rotate the next write schedule (only registered every WR_ROT_PERIOD)
	// This is reset every SIMD rows written
	always_comb begin : writeBankScheduleRotation
		// Reset the write bank allocation after SIMD Rows
		if (wr_bank_reset == J-1)
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
	logic[$clog2(WR_ROT_PERIOD)-1:0] wr_rot_counter;
	logic[$clog2(I*J/SIMD)-1:0]      wr_counter;

	// Bank schedule reset (Resets the bank write after SIMD*I elements written)
	logic[$clog2(J)-1:0]      	 wr_bank_reset;

	always_ff @(posedge clk) begin
		if (rst) begin
			for(int unsigned i=0; i<SIMD; i++) wr_bank_schedule[i] <= i;
			wr_rot_counter <= 'd0;
			wr_counter <= 'd0;
			wr_bank_reset <= 'd0;
		end
		else 
			if (ivld && irdy) begin // Detect once we need to rotate and perform right rotation
		
				if(wr_bank_reset == J-1) 
					wr_bank_reset <= 'd0;
				else
					wr_bank_reset <= wr_bank_reset + 'd1;

				if (wr_rot_counter == WR_ROT_PERIOD - 1) begin
					wr_rot_counter <= 'd0;
					if (wr_counter == (I*J/SIMD - 1))  
						wr_counter <= 'd0;
					for (int unsigned i = 0; i < SIMD; i++) wr_bank_schedule[i] <= next_wr_bank_schedule[i];
				end	
				else begin 
					wr_rot_counter <= wr_rot_counter + 'd1;
					wr_counter <= wr_counter + 'd1;
				end
			end
	end

	// Job tracking and bank page locking
	logic [1:0] wr_jobs_done; // Bit vector tracking when writes have been completed to pages
	logic rd_page_in_progress; // 0 - reading from PAGE A, 1 - reading from PAGE B  
	logic [$clog2(BANK_DEPTH)-1:0] page_rd_offset;

	always_ff @(posedge clk) begin
		if (rst) begin
			wr_jobs_done <= 2'b00;
			rd_page_in_progress <= 1'b0;
		end

		// Track if we have completed a job
		if (wr_addr == PAGE_OFFSET   - 1) wr_jobs_done[0] <= 1'b1;
		if (wr_addr == 2*PAGE_OFFSET - 1) wr_jobs_done[1] <= 1'b1;

		// Clear the relevant job once it is read
		if ((rd_j_cnt == J-1) && (rd_i_cnt+SIMD == I) && (osb_rdy && osb_vld_d)) begin	
		       wr_jobs_done[rd_page_in_progress] <= 1'b0;	
		       rd_page_in_progress <= !rd_page_in_progress;
		end
	end

	assign page_rd_offset = rd_page_in_progress ? PAGE_OFFSET : 'd0;
        assign irdy = !wr_jobs_done[0] || !wr_jobs_done[1]; 	

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
	logic[$clog2(I)-1 : 0] rd_i_cnt;
	logic[$clog2(J)-1 : 0] rd_j_cnt;
	logic rd_guard;
	assign rd_guard = !rd_page_in_progress && !wr_jobs_done[0] && !wr_jobs_done[1];

	// Logic to track which iteration we are on for the read side
	always_ff @(posedge clk) begin : readIndexLoopTracking
		if (rst) begin
			rd_i_cnt <= 'd0;
			rd_j_cnt <= 'd0;
		end
		else
			if(osb_rdy && !rd_guard) 
				if((rd_i_cnt+SIMD) >= I) begin
					rd_i_cnt <= 'd0;
					if( rd_j_cnt < J-1) 
						rd_j_cnt <= rd_j_cnt + 'd1;
					else 
						rd_j_cnt <= 'd0;
				end
				else
					rd_i_cnt <= rd_i_cnt + SIMD; 
	end : readIndexLoopTracking

	// Combinatorial generation of the current set of Read addresses
	always_comb begin : bankRdAddrGen
		for(int unsigned i=0; i<SIMD; i++) mem_banks_rd_addr[i] = 'd0; // default to avoid latch inference	
		for(int unsigned i=0; i < SIMD; i++) 
			mem_banks_rd_addr[rd_pat[i]] = ((rd_i_cnt + i)*J + rd_j_cnt)/SIMD + page_rd_offset; 
	end : bankRdAddrGen
	// --------------------------------------------------------------------------
	
	// --------------------------------------------------------------------------
        logic [SIMD-1:0][BITS-1:0]     data_reg; // remapped output   
	logic [$clog2(I*J/SIMD)-1:0]   rd_pattern_idx;
	rd_pattern_col_t               rd_pattern_col_ff; // The fowarded rotation pattern 

	// Forward the current RD_PATTERN row onto the next pipeline stage
	always_ff @(posedge clk) begin : rdPatternColForwarding
		if (rst) osb_vld <= 0;
		else begin 
			osb_vld <= !rd_guard;
			osb_vld_d <= osb_vld;
			if (osb_rdy && !rd_guard) 
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
			if (osb_rdy && !rd_guard) begin
				rd_counter <= rd_counter + 'd1;
				if (rd_counter == RD_ROT_PERIOD-1) begin
					rd_counter <= 'd0;
					for(int unsigned i=0; i<SIMD; i++)
						rd_pat[i] <= rd_pat_next[i];
				end 
					
				// At the page boundary reset our RD_PATTERN lookup
				if ((rd_j_cnt == J-1) && (rd_i_cnt+SIMD == I )) begin 
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
		.ivld(osb_vld),
		.irdy(osb_rdy),

		.odat(odat),
		.ovld(ovld),
		.ordy(ordy)
	);
	
endmodule : ptranspose
