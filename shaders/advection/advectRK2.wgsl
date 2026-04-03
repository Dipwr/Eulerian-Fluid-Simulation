struct SimData {
	width: u32,
	height: u32,
	overrelaxation: f32,
	dt: f32,
	scale: f32,
	pressure_sensitivity: f32,
};

@group(0) @binding(0) var<storage, read> vel_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> vel_out: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> packed_mask: array<u32>;
@group(0) @binding(3) var<uniform> sim: SimData;

fn get_u(x: i32, y: i32) -> f32 {
	let idx_x = u32(clamp(x, 0, i32(sim.width) - 1));
	let idx_y = u32(clamp(y, 0, i32(sim.height) - 1));
	return vel_in[idx_x * sim.height + idx_y].x;
}

fn get_v(x: i32, y: i32) -> f32 {
	let idx_x = u32(clamp(x, 0, i32(sim.width) - 1));
	let idx_y = u32(clamp(y, 0, i32(sim.height) - 1));
	return vel_in[idx_x * sim.height + idx_y].y;
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

// Helper to get a full velocity vector at any point
fn sample_velocity(x: f32, y: f32) -> vec2<f32> {
	return vec2<f32>(sample_u(x, y), sample_v(x, y));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let index = id.x;
	if (index >= sim.width * sim.height) { return; }

	let x = index / sim.height;
	let y = index % sim.height;
	let mask = packed_mask[index];
	if ((mask & 1u) == 0u) { return; }

	let center_vel = vel_in[index];
	let sx0 = (mask >> 1u) & 1u; 
	let sy0 = (mask >> 3u) & 1u; 

	var new_u = center_vel.x;
	if (sx0 == 1u) {
		// Location of U-face: (x, y + 0.5)
		let pos = vec2<f32>(f32(x), f32(y) + 0.5);
		let v_start = sample_velocity(pos.x, pos.y);
		
		// 1. Midpoint trace back
		let mid_pos = pos - v_start * (sim.dt * 0.5);
		let v_mid = sample_velocity(mid_pos.x, mid_pos.y);
		
		// 2. Full step back using midpoint velocity
		let prev_pos = pos - v_mid * sim.dt;
		new_u = sample_u(prev_pos.x, prev_pos.y);
	}

	var new_v = center_vel.y;
	if (sy0 == 1u) {
		// Location of V-face: (x + 0.5, y)
		let pos = vec2<f32>(f32(x) + 0.5, f32(y));
		let v_start = sample_velocity(pos.x, pos.y);
		
		let mid_pos = pos - v_start * (sim.dt * 0.5);
		let v_mid = sample_velocity(mid_pos.x, mid_pos.y);
		
		let prev_pos = pos - v_mid * sim.dt;
		new_v = sample_v(prev_pos.x, prev_pos.y);
	}

	vel_out[index] = vec2<f32>(new_u, new_v);
}