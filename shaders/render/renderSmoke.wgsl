struct SimData {
	width: u32,
	height: u32,
	overrelaxation: f32,
	dt: f32,
	scale: f32,
    pressure_sensitivity: f32,
};

@group(0) @binding(0) var<storage, read> smoke: array<f32>;
@group(0) @binding(1) var<uniform> sim: SimData;
@group(0) @binding(2) var<storage, read> packed_mask: array<u32>;

struct VertexOutput {
	@builtin(position) pos: vec4<f32>,
	@location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32) -> VertexOutput {
	// Draw 6 vertices per cell (2 triangles = 1 quad)
	let cell_idx = v_idx / 6u;
	let point_idx = v_idx % 6u;

	let w = f32(sim.width);
	let h = f32(sim.height);
	
	let cx = f32(cell_idx / sim.height);
	let cy = f32(cell_idx % sim.height);

	let cell_w = 2.0 / w;
	let cell_h = 2.0 / h;

	// The 6 points of a Quad (0,0 to 1,1)
	var quad_offsets = array<vec2<f32>, 6>(
		vec2<f32>(0.0, 0.0),
		vec2<f32>(1.0, 0.0),
		vec2<f32>(0.0, 1.0),
		vec2<f32>(0.0, 1.0),
		vec2<f32>(1.0, 0.0),
		vec2<f32>(1.0, 1.0)
	);

	let offset = quad_offsets[point_idx];
	
	let screen_x = (cx / w) * 2.0 - 1.0 + (offset.x * cell_w);
	let screen_y = 1.0 - (cy / h) * 2.0 - (offset.y * cell_h); // Flip Y

	let density = 1-smoke[cell_idx];

	var out: VertexOutput;
	out.pos = vec4<f32>(screen_x, screen_y, 0.0, 1.0);
	
	let mask = packed_mask[cell_idx];
	if ((mask & 1u) == 0u) {
		out.color = vec4<f32>(0.2, 0.3, 0.5, 1.0); 
	} else {
		let density = 1-smoke[cell_idx];
		out.color = vec4<f32>(density, density, density, 1.0);
	}
	
	return out;
}

@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
	return color;
}