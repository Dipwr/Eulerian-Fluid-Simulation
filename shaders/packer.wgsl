struct SimData {
	width: u32,
	height: u32,
	overrelaxation: f32,
	dt: f32,
	scale: f32,
    pressure_sensitivity: f32,
};

@group(0) @binding(0) var<storage, read> solid: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed_mask: array<u32>;
@group(0) @binding(2) var<uniform> sim: SimData;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let index = id.x;
	if (index >= sim.width * sim.height) { return; }

	let n = sim.height;
	let x = index / n;
	let y = index % n;

	let s_center = u32(solid[index]);
	
	// FIXED: Prevent memory wrapping & out-of-bounds reads.
	// Default outside boundaries to 1u (Fluid) so open air doesn't bounce pressure!
	var sx0: u32 = 1u; 
	if (x > 0u) { sx0 = u32(solid[index - n]); }
	
	var sx1: u32 = 1u; 
	if (x < sim.width - 1u) { sx1 = u32(solid[index + n]); }
	
	var sy0: u32 = 1u; 
	if (y > 0u) { sy0 = u32(solid[index - 1u]); }
	
	var sy1: u32 = 1u; 
	if (y < sim.height - 1u) { sy1 = u32(solid[index + 1u]); }

	var mask: u32 = 0u;
	mask |= (s_center << 0u);
	mask |= (sx0 << 1u);
	mask |= (sx1 << 2u);
	mask |= (sy0 << 3u);
	mask |= (sy1 << 4u);

	packed_mask[index] = mask;
}