struct SimData {
	width: u32,
	height: u32,
	overrelaxation: f32,
	dt: f32,
	scale: f32,
    pressure_sensitivity: f32,
};

@group(0) @binding(0) var<storage, read> hi_div: array<f32>;
@group(0) @binding(1) var<storage, read_write> lo_div: array<f32>;
@group(0) @binding(2) var<storage, read> hi_mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> lo_mask: array<u32>;
@group(0) @binding(4) var<uniform> sim: SimData;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let lo_x = id.x;
	let lo_y = id.y;
	let lo_width = sim.width / 2u;
	let lo_height = sim.height / 2u;

	if (lo_x >= lo_width || lo_y >= lo_height) { return; }

	let hi_x = lo_x * 2u;
	let hi_y = lo_y * 2u;
	let lo_idx = lo_x * lo_height + lo_y;

	// --- 1. Downsample Divergence (Average 4 cells) ---
	var sum = hi_div[hi_x * sim.height + hi_y];
	sum += hi_div[(hi_x + 1u) * sim.height + hi_y];
	sum += hi_div[hi_x * sim.height + (hi_y + 1u)];
	sum += hi_div[(hi_x + 1u) * sim.height + (hi_y + 1u)];
	lo_div[lo_idx] = sum / 4.0;

    // --- 2. Downsample Packed Mask ---
	let m00 = hi_mask[hi_x * sim.height + hi_y];
	let m10 = hi_mask[(hi_x + 1u) * sim.height + hi_y];
	let m01 = hi_mask[hi_x * sim.height + (hi_y + 1u)];
	let m11 = hi_mask[(hi_x + 1u) * sim.height + (hi_y + 1u)];

	// CHANGE: Only fluid (bit 0) if ALL 4 high-res cells are fluid (AND instead of OR)
	var final_mask = (m00 & m10 & m01 & m11) & 1u;

	// The boundary logic (bits 1-4) should also be strict
	final_mask |= ((m00 & 2u) & (m01 & 2u));  // Left face open if both hi-res lefts are open
	final_mask |= ((m10 & 4u) & (m11 & 4u));  // Right face open
	final_mask |= ((m00 & 8u) & (m10 & 8u));  // Top face open
	final_mask |= ((m01 & 16u) & (m11 & 16u)); // Bottom face open

	lo_mask[lo_idx] = final_mask;
}