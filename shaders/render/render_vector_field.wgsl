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

// Change this to sample every Nth cell!
const SPACING: u32 = 10u; 

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32) -> VertexOutput {
	// 2 vertices per line
	let cell_idx = v_idx / 2u;
	let point_idx = v_idx % 2u; 

	let w = sim.width;
	let h = sim.height;

	let cx = cell_idx / h;
	let cy = cell_idx % h;

	// --- THE SPACING TRICK ---
	// If this cell isn't on our spacing grid, return a zero-length line to hide it
	if (cx % SPACING != 0u || cy % SPACING != 0u) {
		return VertexOutput(vec4<f32>(0.0), vec4<f32>(0.0)); 
	}

	// 1. Calculate the Centered Velocity
	// Clamp the lookups so they never cross column boundaries!
	let right_x = min(cx + 1u, w - 1u);
	let bottom_y = min(cy + 1u, h - 1u);
	
	let right_idx = right_x * h + cy;
	let bottom_idx = cx * h + bottom_y;

	let center_u = (vel[cell_idx].x + vel[right_idx].x) * 0.5;
	let center_v = (vel[cell_idx].y + vel[bottom_idx].y) * 0.5;
	let center_vel = vec2<f32>(center_u, center_v);

	// 2. Calculate Screen Position (Center of the cell)
	let cell_w = 2.0 / f32(w);
	let cell_h = 2.0 / f32(h);

	let screen_x = (f32(cx) / f32(w)) * 2.0 - 1.0 + (cell_w * 0.5);
	let screen_y = 1.0 - (f32(cy) / f32(h)) * 2.0 - (cell_h * 0.5);

	var pos = vec2<f32>(screen_x, screen_y);

	// 3. Draw the Vector
	if (point_idx == 1u) {
		// Multiply by spacing so the arrows grow proportionately to the grid gaps
		pos.x += center_vel.x * sim.scale * f32(SPACING);
		pos.y -= center_vel.y * sim.scale * f32(SPACING); // minus because Y goes up
	}

	// 4. Color it based on speed (Cyan / Blue)
	let speed = length(center_vel);
	let intensity = clamp(speed * 1.5 + 0.2, 0.2, 1.0);
	let color = vec4<f32>(0.1, intensity * 0.8, intensity, 1.0);

	var out: VertexOutput;
	out.pos = vec4<f32>(pos, 0.0, 1.0);
	out.color = color;
	return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	return in.color;
}