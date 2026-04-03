struct SimData {
	width: u32,
	height: u32,
	overrelaxation: f32,
	dt: f32,
	scale: f32,
    pressure_sensitivity: f32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) cell_idx: u32,
};

@group(0) @binding(0) var<storage, read> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> packed_mask: array<u32>;
@group(0) @binding(2) var<uniform> sim: SimData;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
	let cell_idx = vid / 6u;
	let tri_idx = vid % 6u;

	let x = f32(cell_idx / sim.height);
	let y = f32(cell_idx % sim.height);

	var pos = array<vec2<f32>, 6>(
		vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
		vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
	);

	let world_x = (x + pos[tri_idx].x) / f32(sim.width) * 2.0 - 1.0;
	let world_y = ((y + pos[tri_idx].y) / f32(sim.height) * 2.0 - 1.0) * -1.0;

	return VertexOutput(vec4<f32>(world_x, world_y, 0.0, 1.0), cell_idx);
}

fn getSciColor(val: f32, minVal: f32, maxVal: f32) -> vec3<f32> {
	let d = maxVal - minVal;
	let v = clamp((val - minVal) / d, 0.0, 1.0-0.000001);
	let m = 0.25;
	let num = u32(v / m);
	let s = (v - f32(num) * m) / m;
	
	var r = 0.0;
	var g = 0.0;
	var b = 0.0;
	
	if (num == 0u) {
		r = 0.0; g = s; b = 1.0;
	} else if (num == 1u) {
		r = 0.0; g = 1.0; b = 1.0 - s;
	} else if (num == 2u) {
		r = s; g = 1.0; b = 0.0;
	} else {
		r = 1.0; g = 1.0 - s; b = 0.0;
	}
	
	return vec3<f32>(r, g, b);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let mask = packed_mask[in.cell_idx];
    
    // Obstacles are black
    if ((mask & 1u) == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let p = pressure[in.cell_idx];
    
    // Use the uniform to set the range. 
    // Higher sensitivity makes subtle pressure changes more visible.
    let min_p = -sim.pressure_sensitivity; 
    let max_p = sim.pressure_sensitivity;
    
    let color = getSciColor(p, min_p, max_p);
    return vec4<f32>(color, 1.0);
}