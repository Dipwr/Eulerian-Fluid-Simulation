struct SimData {
	width: u32,
	height: u32,
	overrelaxation: f32,
	dt: f32,
	scale: f32,
    pressure_sensitivity: f32,
};

@group(0) @binding(0) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> smoke_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> smoke_out: array<f32>;
@group(0) @binding(3) var<storage, read> packed_mask: array<u32>;
@group(0) @binding(4) var<uniform> sim: SimData;

fn get_smoke(x: i32, y: i32) -> f32 {
	let idx_x = u32(clamp(x, 0, i32(sim.width) - 1));
	let idx_y = u32(clamp(y, 0, i32(sim.height) - 1));
	return smoke_in[idx_x * sim.height + idx_y];
}

fn sample_smoke(x: f32, y: f32) -> f32 {
	// Keep within bounds
	let clamp_x = clamp(x, 0.0, f32(sim.width) - 1.0);
	let clamp_y = clamp(y, 0.0, f32(sim.height) - 1.0);

	let x0 = i32(clamp_x);
	let y0 = i32(clamp_y);
	let tx = clamp_x - f32(x0);
	let ty = clamp_y - f32(y0);

	// Bilinear interpolation
	let top = mix(get_smoke(x0, y0), get_smoke(x0 + 1, y0), tx);
	let bot = mix(get_smoke(x0, y0 + 1), get_smoke(x0 + 1, y0 + 1), tx);
	return mix(top, bot, ty);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let index = id.x;
	if (index >= sim.width * sim.height) { return; }

	let x = index / sim.height;
	let y = index % sim.height;

	let mask = packed_mask[index];
	if ((mask & 1u) == 0u) {
		smoke_out[index] = 0.0;
		return;
	}

	// Average the staggered face velocities to get the center velocity
	let right_idx = index + sim.height;
	let bottom_idx = index + 1u;
	
    let center = vel[index];

	let center_u = (center.x + vel[right_idx].x) * 0.5;
	let center_v = (center.y + vel[bottom_idx].y) * 0.5;

	// Trace backwards in time
	let prev_x = f32(x) - center_u * sim.dt;
	let prev_y = f32(y) - center_v * sim.dt;

	// Sample the previous smoke density and write it
	smoke_out[index] = sample_smoke(prev_x, prev_y);
}