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

fn get_u(x: i32, y: i32) -> f32 {
	let idx_x = u32(clamp(x, 0, i32(sim.width) - 1));
	let idx_y = u32(clamp(y, 0, i32(sim.height) - 1));
	return vel[idx_x * sim.height + idx_y].x;
}

fn get_v(x: i32, y: i32) -> f32 {
	let idx_x = u32(clamp(x, 0, i32(sim.width) - 1));
	let idx_y = u32(clamp(y, 0, i32(sim.height) - 1));
	return vel[idx_x * sim.height + idx_y].y;
}

fn sample_velocity(x: f32, y: f32) -> vec2<f32> {
	// Sample center U (average of left/right)
	let u = (sample_u(x, y) + sample_u(x + 1.0, y)) * 0.5;
	// Sample center V (average of top/bottom)
	let v = (sample_v(x, y) + sample_v(x, y + 1.0)) * 0.5;
	return vec2<f32>(u, v);
}

fn sample_u(x: f32, y: f32) -> f32 {
	let clamp_x = clamp(x, 0.0, f32(sim.width) - 1.0);
	let clamp_y = clamp(y - 0.5, 0.0, f32(sim.height) - 1.0); 
	let x0 = i32(clamp_x);
	let y0 = i32(clamp_y);
	let tx = clamp_x - f32(x0);
	let ty = clamp_y - f32(y0);
	let top = mix(get_u(x0, y0), get_u(x0 + 1, y0), tx);
	let bot = mix(get_u(x0, y0 + 1), get_u(x0 + 1, y0 + 1), tx);
	return mix(top, bot, ty);
}

fn sample_v(x: f32, y: f32) -> f32 {
	let clamp_x = clamp(x - 0.5, 0.0, f32(sim.width) - 1.0);
	let clamp_y = clamp(y, 0.0, f32(sim.height) - 1.0);
	let x0 = i32(clamp_x);
	let y0 = i32(clamp_y);
	let tx = clamp_x - f32(x0);
	let ty = clamp_y - f32(y0);
	let top = mix(get_v(x0, y0), get_v(x0 + 1, y0), tx);
	let bot = mix(get_v(x0, y0 + 1), get_v(x0 + 1, y0 + 1), tx);
	return mix(top, bot, ty);
}

fn get_smoke(x: i32, y: i32) -> f32 {
	let idx_x = u32(clamp(x, 0, i32(sim.width) - 1));
	let idx_y = u32(clamp(y, 0, i32(sim.height) - 1));
	return smoke_in[idx_x * sim.height + idx_y];
}

fn sample_smoke(x: f32, y: f32) -> f32 {
	let clamp_x = clamp(x, 0.0, f32(sim.width) - 1.0);
	let clamp_y = clamp(y, 0.0, f32(sim.height) - 1.0);
	let x0 = i32(clamp_x);
	let y0 = i32(clamp_y);
	let tx = clamp_x - f32(x0);
	let ty = clamp_y - f32(y0);
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

	// Current center position
	let pos = vec2<f32>(f32(x), f32(y));
	
	// Center velocity
	let v_start = vec2<f32>(
		(vel[index].x + vel[index + sim.height].x) * 0.5,
		(vel[index].y + vel[index + 1u].y) * 0.5
	);

	// RK2 Steps
	let mid_pos = pos - v_start * (sim.dt * 0.5);
	let v_mid = sample_velocity(mid_pos.x, mid_pos.y);
	let prev_pos = pos - v_mid * sim.dt;

	smoke_out[index] = sample_smoke(prev_pos.x, prev_pos.y);
}