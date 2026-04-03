struct SimData {
	width: u32,
	height: u32,
	overrelaxation: f32,
	dt: f32,
	scale: f32,
    pressure_sensitivity: f32,
};

@group(0) @binding(0) var<storage, read_write> vel: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> pressure: array<f32>;
@group(0) @binding(2) var<storage, read> packed_mask: array<u32>;
@group(0) @binding(3) var<uniform> sim: SimData;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let idx = id.x;
	if (idx >= sim.width * sim.height) { return; }

	let x = idx / sim.height;
	let y = idx % sim.height;

	if (x == 0u || y == 0u) { return; }

	let p_center = pressure[idx];
	let p_left = pressure[idx - sim.height];
	let p_top = pressure[idx - 1u];

	let mask = packed_mask[idx];
	let sx0 = f32((mask >> 1u) & 1u);
	let sy0 = f32((mask >> 3u) & 1u);

	// u = u - dp/dx
	// v = v - dp/dy
	vel[idx].x -= (p_center - p_left) * sx0;
	vel[idx].y -= (p_center - p_top) * sy0;
}