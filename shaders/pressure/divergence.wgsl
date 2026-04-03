struct SimData {
	width: u32,
	height: u32,
	overrelaxation: f32,
	dt: f32,
	scale: f32,
    pressure_sensitivity: f32,
};

@group(0) @binding(0) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> div: array<f32>;
@group(0) @binding(2) var<storage, read> packed_mask: array<u32>;
@group(0) @binding(3) var<uniform> sim: SimData;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let idx = id.x;
	if (idx >= sim.width * sim.height) { return; }

	let x = idx / sim.height;
	let y = idx % sim.height;
	
	// Boundary check
	if (x == 0u || x >= sim.width - 1u || y == 0u || y >= sim.height - 1u) { return; }

	let mask = packed_mask[idx];
	if ((mask & 1u) == 0u) { return; }

	let right_idx = idx + sim.height;
	let bottom_idx = idx + 1u;

	// Standard MAC Grid Divergence
	let divergence = (vel[right_idx].x - vel[idx].x) + (vel[bottom_idx].y - vel[idx].y);
	div[idx] = divergence;
}