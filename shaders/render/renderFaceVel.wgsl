struct SimData {
	width: u32,
	height: u32,
	overrelaxation: f32,
	dt: f32,
	scale: f32,
    pressure_sensitivity: f32,
};

@group(0) @binding(0) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> sim: SimData;

struct VertexOutput {
	@builtin(position) pos: vec4<f32>,
	@location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32) -> VertexOutput {
	// We draw 4 vertices per cell (2 lines)
	let cell_idx = v_idx / 4u;
	let point_idx = v_idx % 4u; // 0,1 = U line | 2,3 = V line

	let w = f32(sim.width);
	let h = f32(sim.height);
	
	// Column-major mapping
	let cx = f32(cell_idx / sim.height);
	let cy = f32(cell_idx % sim.height);

	let cell_w = 2.0 / w;
	let cell_h = 2.0 / h;

	// Base Top-Left coordinate in NDC (-1.0 to 1.0 screen space)
	let base_x = (cx / w) * 2.0 - 1.0;
	let base_y = 1.0 - (cy / h) * 2.0;

	let v = vel[cell_idx];
	var pos = vec2<f32>(0.0);
	var color = vec3<f32>(0.0);
	
	// A multiplier to make the lines visible. Tweak this!
	let scale = sim.scale;

	if (point_idx == 0u || point_idx == 1u) {
		// --- LEFT FACE (U Velocity) ---
		// Anchor it halfway down the left side
		pos = vec2<f32>(base_x, base_y - (cell_h / 2.0));
		
		// Color it Red, dim it if the velocity is near zero
		let intensity = clamp(abs(v.x) * 2.0 + 0.2, 0.1, 1.0);
		color = vec3<f32>(intensity, 0.1, 0.1); 
		
		if (point_idx == 1u) {
			pos.x += v.x * scale; // Extend the line based on velocity
		}
	} else {
		// --- TOP FACE (V Velocity) ---
		// Anchor it halfway across the top
		pos = vec2<f32>(base_x + (cell_w / 2.0), base_y);
		
		// Color it Green
		let intensity = clamp(abs(v.y) * 2.0 + 0.2, 0.1, 1.0);
		color = vec3<f32>(0.1, intensity, 0.1);
		
		if (point_idx == 3u) {
			pos.y -= v.y * scale; // Minus because screen Y goes UP
		}
	}

	var out: VertexOutput;
	out.pos = vec4<f32>(pos, 0.0, 1.0);
	out.color = vec4<f32>(color, 1.0);
	return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	return in.color;
}