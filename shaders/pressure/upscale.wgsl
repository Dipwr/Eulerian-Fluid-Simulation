struct SimData {
	width: u32,
	height: u32,
	overrelaxation: f32,
	dt: f32,
	scale: f32,
    pressure_sensitivity: f32,
};

@group(0) @binding(0) var<storage, read> lo_p: array<f32>;
@group(0) @binding(1) var<storage, read_write> hi_p: array<f32>;
@group(0) @binding(2) var<uniform> sim: SimData; // High-res dimensions
@group(0) @binding(3) var<storage, read> packed_mask: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let idx = id.x;
	if (idx >= sim.width * sim.height) { return; }

	// 1. Calculate high-resolution coordinates
	let hi_x = idx / sim.height;
	let hi_y = idx % sim.height;

	// 2. Calculate low-resolution coordinates and dimensions
	let lo_x = hi_x / 2u;
	let lo_y = hi_y / 2u;
	let lo_height = sim.height / 2u;
	let lo_width = sim.width / 2u;

	// 3. Calculate fractional offsets (0.0 or 0.5) for interpolation
	let fx = f32(hi_x % 2u) * 0.5;
	let fy = f32(hi_y % 2u) * 0.5;

	// 4. Safely fetch the 4 neighboring low-res pixels
	// We use min() to prevent out-of-bounds errors at the far edges
	let x0 = lo_x;
	let x1 = min(lo_x + 1u, lo_width - 1u);
	let y0 = lo_y;
	let y1 = min(lo_y + 1u, lo_height - 1u);

	let p00 = lo_p[x0 * lo_height + y0];
	let p10 = lo_p[x1 * lo_height + y0];
	let p01 = lo_p[x0 * lo_height + y1];
	let p11 = lo_p[x1 * lo_height + y1];

	// 5. Perform Bilinear Interpolation
	let p_smooth = mix(
		mix(p00, p10, fx),
		mix(p01, p11, fx),
		fy
	);

	let hi_mask_val = packed_mask[idx];
	if ((hi_mask_val & 1u) == 0u) {
		hi_p[idx] = 0.0; // Keep solids at zero pressure
	} else {
		hi_p[idx] = p_smooth; // Only apply jumpstart to fluid
	}
}