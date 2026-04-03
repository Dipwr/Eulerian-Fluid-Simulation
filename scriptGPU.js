let device;
let velocityBuffer, velOutBuffer, smokeBuffer, smokeOutBuffer, solidBuffer, packedMaskBuffer, simUniforms, redUniformBuffer, blackUniformBuffer, stagingBuffer, pressureBuffer, divergenceBuffer, loResDivergenceBuffer, loResPressureBuffer, loSimUniforms, loResMaskBuffer;
let advectPipeline, incompressiblePipeline, advectSmokePipeline, packerPipeline, smokeRenderPipeline, velocityRenderPipeline, gridPipeline, vectorFieldPipeline, divergencePipeline, pressurePipeline, subtractPipeline, downsamplePipeline, upscalePipeline, pressureRenderPipeline;
let divergenceBindGroup, pressRedBindGroup, pressBlackBindGroup, subtractBindGroup, advectBindGroup, advectSmokeBindGroup, packerBindGroup, smokeRenderBindGroup, velocityRenderBindGroup, gridBindGroup, vectorFieldBindGroup, downsampleBindGroup, upscaleBindGroup, loPressRedBindGroup, loPressBlackBindGroup, pressureRenderBindGroup;

const canvas = document.querySelector('canvas');
let context = canvas.getContext('webgpu');
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

const gridWidth = 700; 	//even
const gridHeight = 350;	//even

const dpr = window.devicePixelRatio || 1;
canvas.width = gridWidth * dpr;
canvas.height = gridHeight * dpr;

const loWidth = gridWidth / 2;
const loHeight = gridHeight / 2;

const gridSize = gridWidth * gridHeight;

let solidData = new Float32Array(gridWidth * gridHeight).fill(1.0);

const stepsPerFrame = 2; 
const simDt = 1/(60*stepsPerFrame);
const overrelax = 1.7;
const loOverrelax = 1.3;

const visScale = 0.0001;
const pressure_sen = 40;

const workgroups = Math.ceil((gridWidth * gridHeight) / 64);
const incomWorkgroups = Math.ceil((gridWidth * gridHeight/2) / 64);

let obstaclesNeedUpdate = true;

let incomIter = 250;
let loIncomIter = 200;

let lastTime;
let frameCount = 0;
const fpsElement = document.getElementById('fps-counter');

let currentRenderMode = "s"; //s, v, p

let showVectorField = false;

let isLeftDown = false;
let isRightDown = false;
const brushRadius = 15.0;

// --- Wind Tunnel Setup ---
const fanHeight = gridHeight - 2;   
const smokeHeight = Math.floor(gridHeight/10);  

const fanWidth = 2;
const smokeWidth = 2;

const fanX = 0;     
const smokeX = 0;

// Y Positions (Centered)
const fanY = Math.floor(gridHeight / 2) - Math.floor(fanHeight / 2);
const smokeY = Math.floor(gridHeight / 2) - Math.floor(smokeHeight / 2);

const windVelocities = new Float32Array(fanHeight * 2);
for (let i = 0; i < fanHeight; i++) {
	windVelocities[i * 2] = 100.0;     // U Velocity
	windVelocities[i * 2 + 1] = 0.0;  // V Velocity
}

const windSmoke = new Float32Array(smokeHeight);
for (let i = 0; i < smokeHeight; i++) {
	windSmoke[i] = 1.0;
}

async function init() {
	// --- A. The Handshake ---
	const adapter = await navigator.gpu.requestAdapter();
	device = await adapter.requestDevice();

	context.configure({
		device: device,
		format: presentationFormat,
		alphaMode: 'premultiplied',
	});

	// --- B. Memory Allocation (Buffers) ---
	stagingBuffer = device.createBuffer({
		size: gridWidth * gridHeight * 2 * 4, // 2 floats (u,v) * 4 bytes
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	});

	const velocityData = new Float32Array(gridWidth * gridHeight * 2);
	velocityData[(2 * gridHeight + 2)*2] = 10;
	velocityBuffer = device.createBuffer({
		size: velocityData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
	});
	// Upload initial data to the GPU
	device.queue.writeBuffer(velocityBuffer, 0, velocityData);

	velOutBuffer = device.createBuffer({
		size: gridWidth * gridHeight * 2 * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
	});

	const smokeData = new Float32Array(gridWidth * gridHeight);
	smokeData[2 * gridHeight + 2] = 1.0; 
	smokeData[2 * gridHeight + 3] = 1.0;

	smokeBuffer = device.createBuffer({
		size: smokeData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
	});
	device.queue.writeBuffer(smokeBuffer, 0, smokeData);

	smokeOutBuffer = device.createBuffer({
		size: smokeData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
	});

	const solidData = new Float32Array(gridWidth * gridHeight); // 1.0 = fluid, 0.0 = solid
	solidData.fill(1.0);
	solidBuffer = device.createBuffer({
		size: solidData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	});
	device.queue.writeBuffer(solidBuffer, 0, solidData);

	packedMaskBuffer = device.createBuffer({
		size: gridWidth * gridHeight * 4, // u32 is 4 bytes
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});

	loResMaskBuffer = device.createBuffer({
		size: (gridWidth / 2) * (gridHeight / 2) * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	});

	// Pressure field (scalar: 1 float per cell)
	pressureBuffer = device.createBuffer({
		size: gridSize * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});

	// Divergence field (scalar: 1 float per cell)
	divergenceBuffer = device.createBuffer({
		size: gridSize * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
	});

	loResDivergenceBuffer = device.createBuffer({
		size: loWidth * loHeight * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	});

	loResPressureBuffer = device.createBuffer({
		size: loWidth * loHeight * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
	});

	redUniformBuffer = device.createBuffer({
		size: 4,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});
	device.queue.writeBuffer(redUniformBuffer, 0, new Uint32Array([0]));

	blackUniformBuffer = device.createBuffer({
		size: 4,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});
	device.queue.writeBuffer(blackUniformBuffer, 0, new Uint32Array([1]));

	simUniforms = device.createBuffer({
		size: 32, // width(4), height(4), overrelaxation(4), dt(4), scale(4), pressure_sen(4)
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});

	loSimUniforms = device.createBuffer({
		size: 32,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});

	// --- C. Compile Pipelines ---
	// Fetch and compile your WGSL files
	const divergenceShaderCode = await fetch('shaders/pressure/divergence.wgsl').then(r => r.text());
	divergencePipeline = device.createComputePipeline({
		layout: 'auto',
		compute: {
			module: device.createShaderModule({ code: divergenceShaderCode }),
			entryPoint: 'main',
		},
	});

	const pressureShaderCode = await fetch('shaders/pressure/pressure.wgsl').then(r => r.text());
	pressurePipeline = device.createComputePipeline({
		layout: 'auto',
		compute: {
			module: device.createShaderModule({ code: pressureShaderCode }),
			entryPoint: 'main',
		},
	});

	const subtractShaderCode = await fetch('shaders/pressure/subtract.wgsl').then(r => r.text());
	subtractPipeline = device.createComputePipeline({
		layout: 'auto',
		compute: {
			module: device.createShaderModule({ code: subtractShaderCode }),
			entryPoint: 'main',
		},
	});

	const downsampleShaderCode = await fetch('shaders/pressure/downsample.wgsl').then(r => r.text());
	downsamplePipeline = device.createComputePipeline({
		layout: 'auto',
		compute: {
			module: device.createShaderModule({ code: downsampleShaderCode }),
			entryPoint: 'main',
		},
	});

	const upscaleShaderCode = await fetch('shaders/pressure/upscale.wgsl').then(r => r.text());
	upscalePipeline = device.createComputePipeline({
		layout: 'auto',
		compute: {
			module: device.createShaderModule({ code: upscaleShaderCode }),
			entryPoint: 'main',
		},
	});

	const advectShader = await fetch('shaders/advection/advectRK2.wgsl').then(r => r.text());
	advectPipeline = device.createComputePipeline({
		layout: 'auto',
		compute: {
			module: device.createShaderModule({ code: advectShader }),
			entryPoint: 'main',
		},
	});

	const advectSmokeShader = await fetch('shaders/advection/advectSmokeRK2.wgsl').then(r => r.text());
	advectSmokePipeline = device.createComputePipeline({
		layout: 'auto',
		compute: {
			module: device.createShaderModule({ code: advectSmokeShader }),
			entryPoint: 'main',
		},
	});

	const packerShader = await fetch('shaders/packer.wgsl').then(r => r.text());
	packerPipeline = device.createComputePipeline({
		layout: 'auto',
		compute: {
			module: device.createShaderModule({ code: packerShader }),
			entryPoint: 'main',
		},
	});

	const smokeShaderCode = await fetch('shaders/render/renderSmoke.wgsl').then(r => r.text());
	smokeRenderPipeline = device.createRenderPipeline({
		layout: 'auto',
		vertex: {
			module: device.createShaderModule({ code: smokeShaderCode }),
			entryPoint: 'vs_main',
		},
		fragment: {
			module: device.createShaderModule({ code: smokeShaderCode }),
			entryPoint: 'fs_main',
			targets: [{ format: presentationFormat }],
		},
		primitive: { topology: 'triangle-list' }, // Quads need triangles
	});

	const velocityShaderCode = await fetch('shaders/render/renderFaceVel.wgsl').then(r => r.text());
	velocityRenderPipeline = device.createRenderPipeline({
		layout: 'auto',
		vertex: {
			module: device.createShaderModule({ code: velocityShaderCode }),
			entryPoint: 'vs_main',
		},
		fragment: {
			module: device.createShaderModule({ code: velocityShaderCode }),
			entryPoint: 'fs_main',
			targets: [{ format: presentationFormat }],
		},
		primitive: { topology: 'line-list' }, // Arrows need lines
	});

	const gridShader = await fetch('shaders/render/grid.wgsl').then(r => r.text());
	gridPipeline = device.createRenderPipeline({
		layout: 'auto',
		vertex: {
			module: device.createShaderModule({ code: gridShader }),
			entryPoint: 'vs_main',
		},
		fragment: {
			module: device.createShaderModule({ code: gridShader }),
			entryPoint: 'fs_main',
			targets: [{ 
				format: presentationFormat,
				// Optional: Add blending if you want the grid to look softer
				blend: {
					color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
					alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
				}
			}],
		},
		primitive: {
			topology: 'line-list', 
		},
	});

	const vectorShaderCode = await fetch('shaders/render/render_vector_field.wgsl').then(r => r.text());
	vectorFieldPipeline = device.createRenderPipeline({
		layout: 'auto',
		vertex: {
			module: device.createShaderModule({ code: vectorShaderCode }),
			entryPoint: 'vs_main',
		},
		fragment: {
			module: device.createShaderModule({ code: vectorShaderCode }),
			entryPoint: 'fs_main',
			targets: [{ format: presentationFormat }],
		},
		primitive: { topology: 'line-list' }, 
	});

	const pressureRenderShaderCode = await fetch('shaders/render/renderPressure.wgsl').then(r => r.text());
	pressureRenderPipeline = device.createRenderPipeline({
		layout: 'auto',
		vertex: {
			module: device.createShaderModule({ code: pressureRenderShaderCode }),
			entryPoint: 'vs_main',
		},
		fragment: {
			module: device.createShaderModule({ code: pressureRenderShaderCode }),
			entryPoint: 'fs_main',
			targets: [{ format: presentationFormat }],
		},
		primitive: { topology: 'triangle-list' },
	});

	// --- D. Wire Bind Groups ---
	// Divergence Bind Group
	divergenceBindGroup = device.createBindGroup({
		layout: divergencePipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: velocityBuffer } },
			{ binding: 1, resource: { buffer: divergenceBuffer } },
			{ binding: 2, resource: { buffer: packedMaskBuffer } },
			{ binding: 3, resource: { buffer: simUniforms } }
		]
	});

	// Pressure Solver Bind Groups (Red/Black)
	pressRedBindGroup = device.createBindGroup({
		layout: pressurePipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: pressureBuffer } },
			{ binding: 1, resource: { buffer: divergenceBuffer } },
			{ binding: 2, resource: { buffer: packedMaskBuffer } },
			{ binding: 3, resource: { buffer: simUniforms } },
			{ binding: 4, resource: { buffer: redUniformBuffer } }
		]
	});

	pressBlackBindGroup = device.createBindGroup({
		layout: pressurePipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: pressureBuffer } },
			{ binding: 1, resource: { buffer: divergenceBuffer } },
			{ binding: 2, resource: { buffer: packedMaskBuffer } },
			{ binding: 3, resource: { buffer: simUniforms } },
			{ binding: 4, resource: { buffer: blackUniformBuffer } }
		]
	});

	// Subtract Bind Group
	subtractBindGroup = device.createBindGroup({
		layout: subtractPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: velocityBuffer } },
			{ binding: 1, resource: { buffer: pressureBuffer } },
			{ binding: 2, resource: { buffer: packedMaskBuffer } },
			{ binding: 3, resource: { buffer: simUniforms } }
		]
	});

	// Bind Group for Downsampling
	downsampleBindGroup = device.createBindGroup({
	layout: downsamplePipeline.getBindGroupLayout(0),
	entries: [
		{ binding: 0, resource: { buffer: divergenceBuffer } },
		{ binding: 1, resource: { buffer: loResDivergenceBuffer } },
		{ binding: 2, resource: { buffer: packedMaskBuffer } },    // Input: High-res Mask
		{ binding: 3, resource: { buffer: loResMaskBuffer } },     // Output: Low-res Mask
		{ binding: 4, resource: { buffer: simUniforms } }
	]
});

	// Bind Group for Upscaling
	upscaleBindGroup = device.createBindGroup({
		layout: upscalePipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: loResPressureBuffer } }, // Input: Low-res
			{ binding: 1, resource: { buffer: pressureBuffer } },      // Output: High-res
			{ binding: 2, resource: { buffer: simUniforms } },           // High-res dimensions
			{ binding: 3, resource: { buffer: packedMaskBuffer } }  
		]
	});

	// Low-Resolution Solver Bind Groups (Red/Black)
	// Note: These use 'loSimUniforms' so the shader knows it is working on the 300x150 grid
	loPressRedBindGroup = device.createBindGroup({
		layout: pressurePipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: loResPressureBuffer } },
			{ binding: 1, resource: { buffer: loResDivergenceBuffer } },
			{ binding: 2, resource: { buffer: loResMaskBuffer } },     // USE THE NEW LOW-RES MASK
			{ binding: 3, resource: { buffer: loSimUniforms } },
			{ binding: 4, resource: { buffer: redUniformBuffer } }
		]
	});

	loPressBlackBindGroup = device.createBindGroup({
		layout: pressurePipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: loResPressureBuffer } },
			{ binding: 1, resource: { buffer: loResDivergenceBuffer } },
			{ binding: 2, resource: { buffer: loResMaskBuffer } },
			{ binding: 3, resource: { buffer: loSimUniforms } },
			{ binding: 4, resource: { buffer: blackUniformBuffer } }
		]
	});

	advectBindGroup = device.createBindGroup({
		layout: advectPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: velocityBuffer } },
			{ binding: 1, resource: { buffer: velOutBuffer } },     // Writes here
			{ binding: 2, resource: { buffer: packedMaskBuffer } },
			{ binding: 3, resource: { buffer: simUniforms } }
		]
	});

	advectSmokeBindGroup = device.createBindGroup({
		layout: advectSmokePipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: velocityBuffer } },
			{ binding: 1, resource: { buffer: smokeBuffer } },
			{ binding: 2, resource: { buffer: smokeOutBuffer } },
			{ binding: 3, resource: { buffer: packedMaskBuffer } },
			{ binding: 4, resource: { buffer: simUniforms } }
		]
	});

	packerBindGroup = device.createBindGroup({
		layout: packerPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: solidBuffer } },
			{ binding: 1, resource: { buffer: packedMaskBuffer } },
			{ binding: 2, resource: { buffer: simUniforms } }
		]
	});

	smokeRenderBindGroup = device.createBindGroup({
		layout: smokeRenderPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: smokeBuffer } },
			{ binding: 1, resource: { buffer: simUniforms } },
			{ binding: 2, resource: { buffer: packedMaskBuffer } }
		]
	});

	velocityRenderBindGroup = device.createBindGroup({
		layout: velocityRenderPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: velocityBuffer } },
			{ binding: 1, resource: { buffer: simUniforms } }
		]
	});

	gridBindGroup = device.createBindGroup({
		layout: gridPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: simUniforms } }
		]
	});

	vectorFieldBindGroup = device.createBindGroup({
		// CRITICAL: Get the layout directly from the pipeline you intend to use
		layout: vectorFieldPipeline.getBindGroupLayout(0), 
		entries: [
			{ binding: 0, resource: { buffer: velocityBuffer } },
			{ binding: 1, resource: { buffer: simUniforms } }
		]
	});

	pressureRenderBindGroup = device.createBindGroup({
		layout: pressureRenderPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: pressureBuffer } },
			{ binding: 1, resource: { buffer: packedMaskBuffer } },
			{ binding: 2, resource: { buffer: simUniforms } }
		]
	});

	// --- E. Start the Engine ---
	// Write static uniforms (width, height, overrelaxation)
	const initialUniforms = new Float32Array(8); // Use Float32View for easier indexing
	const u32View = new Uint32Array(initialUniforms.buffer);

	u32View[0] = gridWidth;       // u32
	u32View[1] = gridHeight;      // u32
	initialUniforms[2] = overrelax;       
	initialUniforms[3] = simDt;			  
	initialUniforms[4] = visScale;
	initialUniforms[5] = pressure_sen;
	device.queue.writeBuffer(simUniforms, 0, initialUniforms);

	const loSimData = new ArrayBuffer(32);
	new Uint32Array(loSimData, 0, 2).set([loWidth, loHeight]);
	new Float32Array(loSimData, 8, 3).set([loOverrelax, simDt, 1.0]); // Use 1.0 overrelax for lo-res stability
	device.queue.writeBuffer(loSimUniforms, 0, loSimData);

	setupScene()

	lastTime = performance.now();
	requestAnimationFrame(frame);
}

async function getVelocities() {
	const encoder = device.createCommandEncoder();
	
	// 1. Tell the GPU to copy from your simulation buffer to the staging buffer
	encoder.copyBufferToBuffer(
		velocityBuffer, 0, 
		stagingBuffer, 0, 
		gridWidth * gridHeight * 2 * 4
	);

	device.queue.submit([encoder.finish()]);

	// 2. Map the staging buffer to JS memory
	await stagingBuffer.mapAsync(GPUMapMode.READ);
	
	// 3. Create a view into that memory
	const copyArray = new Float32Array(stagingBuffer.getMappedRange());
	
	// 4. Create a fresh snapshot (because getMappedRange is temporary)
	const results = new Float32Array(copyArray);
	
	// 5. Clean up! You must unmap before the GPU can use this buffer again
	stagingBuffer.unmap();
	
	return results;
}

function setBoundaryWalls(options = {}) {
	// Destructure the options with default values (true if not specified)
	const {
		top = true,
		bottom = true,
		left = true,
		right = true
	} = options;

	for (let x = 0; x < gridWidth; x++) {
		for (let y = 0; y < gridHeight; y++) {
			const index = x * gridHeight + y;

			// Apply solid walls only if the specific flag is true
			if (left && x == 0) {
				solidData[index] = 0.0;
			}
			if (right && x == gridWidth - 1) {
				solidData[index] = 0.0;
			}
			if (top && y == 0) {
				solidData[index] = 0.0;
			}
			if (bottom && y == gridHeight - 1) {
				solidData[index] = 0.0;
			}
		}
	}
}

function addCircle(circleX, circleY, radius, solid) {
	for (let i = 0; i < gridWidth; i++) {
		for (let j = 0; j < gridHeight; j++) {
			const dx = i - circleX;
			const dy = j - circleY;
			if (dx * dx + dy * dy < radius * radius) {
				solidData[i * gridHeight + j] = solid;
			}
		}
	}
}

function clearObstacles() {
	solidData.fill(1.0); // Make everything fluid again
}

function commitObstacles() {
	// Send the updated global array to the GPU buffer
	device.queue.writeBuffer(solidBuffer, 0, solidData);
	// Tell the frame loop to run the packer shader next frame
	obstaclesNeedUpdate = true; 
}

function setupScene() {
	clearObstacles();      // 1. Start fresh
	setBoundaryWalls({top:true, bottom:true, left:true, right:false})
	addCircle(gridWidth/7, gridHeight/2, gridHeight/15); // 3. Add first obstacle
	commitObstacles();     // 5. Update the GPU
}

async function frame(){
	const now = performance.now();
	frameCount++;

	if (now - lastTime >= 1000) {
		fpsElement.innerText = `FPS: ${frameCount}`;
		frameCount = 0;
		lastTime = now;
	}

	const encoder = device.createCommandEncoder();
	for (let s = 0; s < stepsPerFrame; s++) {

		for (let w = 0; w < fanWidth; w++) {
			const currentX = fanX + w;
			if (currentX < gridWidth) {
				const velOffset = (currentX * gridHeight + fanY) * 2 * 4; 
				device.queue.writeBuffer(velocityBuffer, velOffset, windVelocities);
			}
		}
		
		for (let w = 0; w < smokeWidth; w++) {
			const currentX = smokeX + w;
			if (currentX < gridWidth - 1) {
				const smokeOffset = (currentX * gridHeight + smokeY) * 4; 
				device.queue.writeBuffer(smokeBuffer, smokeOffset, windSmoke);
			}
		}

		encoder.clearBuffer(divergenceBuffer);
		encoder.clearBuffer(pressureBuffer);

		if (obstaclesNeedUpdate) {
			const packerPass = encoder.beginComputePass();
			packerPass.setPipeline(packerPipeline);
			packerPass.setBindGroup(0, packerBindGroup);
			packerPass.dispatchWorkgroups(workgroups);
			packerPass.end();
			obstaclesNeedUpdate = false;
		}

		//Advection
		const advectPass = encoder.beginComputePass();
		advectPass.setPipeline(advectPipeline);
		advectPass.setBindGroup(0, advectBindGroup);
		advectPass.dispatchWorkgroups(workgroups);
		advectPass.end();

		const smokePass = encoder.beginComputePass();
		smokePass.setPipeline(advectSmokePipeline);
		smokePass.setBindGroup(0, advectSmokeBindGroup);
		smokePass.dispatchWorkgroups(workgroups);
		smokePass.end();

		encoder.copyBufferToBuffer(
			velOutBuffer, 0,
			velocityBuffer, 0,
			gridWidth * gridHeight * 2 * 4
		);

		encoder.copyBufferToBuffer(
			smokeOutBuffer, 0,
			smokeBuffer, 0,
			gridWidth * gridHeight * 4 // 4 bytes per float
		);

		// DIVERGENCE PASS
		const divPass = encoder.beginComputePass();
		divPass.setPipeline(divergencePipeline);
		divPass.setBindGroup(0, divergenceBindGroup);
		divPass.dispatchWorkgroups(workgroups);
		divPass.end();

		//Downsample
		const dsPass = encoder.beginComputePass();
		dsPass.setPipeline(downsamplePipeline);
		dsPass.setBindGroup(0, downsampleBindGroup);
		dsPass.dispatchWorkgroups(Math.ceil(loWidth/8), Math.ceil(loHeight/8));
		dsPass.end();

		//low res pressure
		encoder.clearBuffer(loResPressureBuffer);
		const loPressPass = encoder.beginComputePass();
		loPressPass.setPipeline(pressurePipeline); // Reuse your existing pressure shader
		for (let i = 0; i < loIncomIter; i++) {
			loPressPass.setBindGroup(0, i % 2 === 0 ? loPressRedBindGroup : loPressBlackBindGroup);
			loPressPass.dispatchWorkgroups(Math.ceil((loWidth * loHeight) / 128));
		}
		loPressPass.end();

		// 5. UPSCALE
		const usPass = encoder.beginComputePass();
		usPass.setPipeline(upscalePipeline);
		usPass.setBindGroup(0, upscaleBindGroup);
		usPass.dispatchWorkgroups(workgroups);
		usPass.end();

		// PRESSURE SOLVER PASS
		const pressPass = encoder.beginComputePass();
		pressPass.setPipeline(pressurePipeline);
		for (let i = 0; i < incomIter; i++) {
			pressPass.setBindGroup(0, i % 2 === 0 ? pressRedBindGroup : pressBlackBindGroup);
			pressPass.dispatchWorkgroups(incomWorkgroups);
		}
		pressPass.end();

		// 4. SUBTRACT PASS
		const subPass = encoder.beginComputePass();
		subPass.setPipeline(subtractPipeline);
		subPass.setBindGroup(0, subtractBindGroup);
		subPass.dispatchWorkgroups(workgroups);
		subPass.end();
	}

	// --- Render Pass ---
	const renderPass = encoder.beginRenderPass({
		colorAttachments: [{
			view: context.getCurrentTexture().createView(),
			clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }, // Dark grey background
			loadOp: 'clear',
			storeOp: 'store',
		}]
	});

	if (currentRenderMode == "s") {
		// Draw Smoke
		renderPass.setPipeline(smokeRenderPipeline);
		renderPass.setBindGroup(0, smokeRenderBindGroup);
		renderPass.draw(6 * gridWidth * gridHeight); 
	} 
	else if (currentRenderMode == "v") {
		// 1. Draw Background Grid First
		renderPass.setPipeline(gridPipeline);
		renderPass.setBindGroup(0, gridBindGroup);
		const totalGridVertices = (gridWidth + gridHeight + 2) * 2;
		renderPass.draw(totalGridVertices);

		// 2. Draw Velocity Vectors over the grid
		renderPass.setPipeline(velocityRenderPipeline);
		renderPass.setBindGroup(0, velocityRenderBindGroup);
		renderPass.draw(4 * gridWidth * gridHeight); 
	}else if (currentRenderMode === "p") {
		renderPass.setPipeline(pressureRenderPipeline);
		renderPass.setBindGroup(0, pressureRenderBindGroup);
		renderPass.draw(6 * gridWidth * gridHeight);
	}

	if (showVectorField) {
		renderPass.setPipeline(vectorFieldPipeline);
		renderPass.setBindGroup(0, vectorFieldBindGroup); 
		renderPass.draw(2 * gridWidth * gridHeight); 
	}
	renderPass.end();

	device.queue.submit([encoder.finish()]);

	requestAnimationFrame(frame);
}

function handleInteraction(e) {
	const rect = canvas.getBoundingClientRect();
	const mX = (e.clientX - rect.left)/(canvas.width/gridWidth);
	const mY = (e.clientY - rect.top)/(canvas.height/gridHeight);

	if (isLeftDown) {
		addCircle(mX, mY, brushRadius, 0.0);
	} else if (isRightDown) {
		addCircle(mX, mY, brushRadius, 1.0);
	}
	
	commitObstacles();
}

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

canvas.addEventListener('mousedown', (e) => {
	if (e.button === 0) isLeftDown = true;
	if (e.button === 2) isRightDown = true;
	handleInteraction(e);
});

window.addEventListener('mouseup', (e) => {
	if (e.button === 0) isLeftDown = false;
	if (e.button === 2) isRightDown = false;
});

canvas.addEventListener('mousemove', (e) => {
	if (isLeftDown || isRightDown) {
		handleInteraction(e);
	}
});

init()