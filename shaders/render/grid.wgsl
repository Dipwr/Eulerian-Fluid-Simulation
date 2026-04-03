struct SimData {
	width: u32,
	height: u32,
	overrelaxation: f32,
	dt: f32,
	scale: f32,
    pressure_sensitivity: f32,
};

@group(0) @binding(0) var<uniform> sim: SimData;

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32) -> @builtin(position) vec4<f32> {
	let w = f32(sim.width);
	let h = f32(sim.height);
	
	// Each line takes 2 vertices. 
	let line_idx = v_idx / 2u;
	
	// 0.0 for the start of the line, 1.0 for the end of the line
	let is_end = f32(v_idx % 2u); 
	
	var pos = vec2<f32>(0.0);

	// First, draw all the Vertical Lines
	if (line_idx <= sim.width) {
		let x_percent = f32(line_idx) / w;
		let screen_x = (x_percent * 2.0) - 1.0;
		
		// Y goes from -1.0 to 1.0
		let screen_y = (is_end * 2.0) - 1.0; 
		
		pos = vec2<f32>(screen_x, screen_y);
	} 
	// Then, draw all the Horizontal Lines
	else {
		let h_line_idx = line_idx - (sim.width + 1u);
		let y_percent = f32(h_line_idx) / h;
		let screen_y = (y_percent * 2.0) - 1.0;
		
		// X goes from -1.0 to 1.0
		let screen_x = (is_end * 2.0) - 1.0;
		
		pos = vec2<f32>(screen_x, screen_y);
	}

	return vec4<f32>(pos, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
	// A faint, dark grey color for the grid lines
	return vec4<f32>(0.2, 0.2, 0.2, 1.0); 
}