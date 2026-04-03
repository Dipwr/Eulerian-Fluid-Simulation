struct SimData {
	width: u32,
	height: u32,
	overrelaxation: f32,
	dt: f32,
	scale: f32,
    pressure_sensitivity: f32,
};

@group(0) @binding(0) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> div: array<f32>;
@group(0) @binding(2) var<storage, read> packed_mask: array<u32>;
@group(0) @binding(3) var<uniform> sim: SimData;
@group(0) @binding(4) var<uniform> passColor: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	let half_height = sim.height / 2u;
	let x = i / half_height;
	let y = (i % half_height) * 2u + ((x + passColor) % 2u);

	if (x == 0u || x >= sim.width - 1u || y == 0u || y >= sim.height - 1u) { return; }

	let idx = x * sim.height + y;
	let mask = packed_mask[idx];
	if ((mask & 1u) == 0u) { return; }

	// Sum of open neighbor faces
	let s = f32((mask >> 1u) & 1u) + f32((mask >> 2u) & 1u) + f32((mask >> 3u) & 1u) + f32((mask >> 4u) & 1u);
	if (s == 0.0) { return; }

	let p_left = pressure[idx - sim.height];
	let p_right = pressure[idx + sim.height];
	let p_top = pressure[idx - 1u];
	let p_bottom = pressure[idx + 1u];

	// Jacobi/Gauss-Seidel update for Pressure
	let new_p = (p_left + p_right + p_top + p_bottom - div[idx]) / s;
	pressure[idx] = mix(pressure[idx], new_p, sim.overrelaxation);
}