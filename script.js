const canvas = document.querySelector("canvas");
const ctx = canvas.getContext("2d");


function Fluid(numX, numY, h){
	this.numX = numX+2;
    this.numY = numY+2;
    this.numCells = this.numX * this.numY;
    this.h = h;

    this.u = new Float32Array(this.numCells);
    this.v = new Float32Array(this.numCells);
    this.m = new Float32Array(this.numCells);
    this.newU = new Float32Array(this.numCells);
    this.newV = new Float32Array(this.numCells);
    this.newM = new Float32Array(this.numCells);
	this.prevU = [new Float32Array(this.numCells),new Float32Array(this.numCells)];
	this.prevV = [new Float32Array(this.numCells),new Float32Array(this.numCells)];
	this.prevM = [new Float32Array(this.numCells),new Float32Array(this.numCells)];

    this.s = new Float32Array(this.numCells);
	this.s.fill(1.0);

	this.p = new Float32Array(this.numCells);

    this.overrelaxation = 1.6;

	this.integrationMethod = "rk2";

	this.showMode = "s";
	
	this.emitters = [];
	
	this.pressureVisSensitivity = 0.001;
	
	this.makeIncompressible = function(iterations) {
		let n = this.numY;
		if (this.showMode == "p") this.p.fill(0.0);
		for (let iter = 0; iter < iterations; iter++) {
			for (let i = 1; i < this.numX - 1; i++) {
				for (let j = 1; j < this.numY - 1; j++) {
					if (this.s[i * n + j] == 0.0) continue;
					
					var sx0 = this.s[(i - 1) * n + j]; // left neighbor
					var sx1 = this.s[(i + 1) * n + j]; // right neighbor
					var sy0 = this.s[i * n + j - 1];   // top neighbor (j-1 is above)
					var sy1 = this.s[i * n + j + 1];   // bottom neighbor (j+1 is below)
					
					var s = sx0 + sx1 + sy0 + sy1;
					if (s == 0.0) continue;
					
					// Divergence: (Right - Left) + (Bottom - Top)
					let div = this.u[(i + 1) * n + j] 
					- this.u[i * n + j] 
					+ this.v[i * n + (j + 1)]
					- this.v[i * n + j];
					
					let correction = -div / s;
					correction *= this.overrelaxation;
					// 3. Accumulate into the pressure field
					if (this.showMode == "p") this.p[i * n + j] += correction;


					// 4. Apply velocity updates as you did before
					this.u[i * n + j] -= correction * sx0;
					this.u[(i + 1) * n + j] += correction * sx1;
					this.v[i * n + j] -= correction * sy0;
					this.v[i * n + (j + 1)] += correction * sy1;     // Bottom: decrease
				}
			}
		}
		this.prevU[1].set(this.u);
		this.prevV[1].set(this.v);
		this.prevM[1].set(this.m);
	}
	
    this.advect = function(dt){
		let n = this.numY;
		this.newU.set(this.u);
    	this.newV.set(this.v);
		this.newM.set(this.m);
		for (let i = 1; i < this.numX - 1; i++) {
			for (let j = 1; j < this.numY - 1; j++) {
				if (this.s[i * n + j] == 0.0) continue;
				
				if (this.showMode == "s") this.advectField(i, j, dt); //M field
				
				if (this.s[(i-1) * n + j] == 1.0){
					this.advectField(i, j, dt, "U");
				}
				
				if (this.s[i * n + (j-1)] == 1.0){
					this.advectField(i, j, dt, "V");
				}
			}
		}
		let temp = this.prevU[0];
		this.prevU[0] = this.prevU[1];
		this.prevU[1] = temp;

		temp = this.prevV[0];
		this.prevV[0] = this.prevV[1];
		this.prevV[1] = temp;

		temp = this.prevM[0];
		this.prevM[0] = this.prevM[1];
		this.prevM[1] = temp;

		this.u.set(this.newU);
		this.v.set(this.newV);
		this.m.set(this.newM);
    }
	
	this.advectField = function(i, j, dt, type) {
		const n = this.numY;
		let x, y, velX, velY;

		if (type == 'U') {
			// Horizontal velocity face
			x = i;
			y = j + 0.5;
			velX = this.u[i * n + j];
			velY = (this.v[(i - 1) * n + j] + this.v[i * n + j] + 
			        this.v[i * n + (j - 1)] + this.v[(i - 1) * n + (j - 1)]) * 0.25;
		} 
		else if (type == 'V') {
			// Vertical velocity face
			x = i + 0.5;
			y = j;
			velX = (this.u[i * n + (j - 1)] + this.u[(i + 1) * n + (j - 1)] + 
			        this.u[(i + 1) * n + j] + this.u[i * n + j]) * 0.25;
			velY = this.v[i * n + j];
		} 
		else { // type === 'M'
			// Cell center (Smoke/Mass)
			x = i + 0.5;
			y = j + 0.5;
			velX = (this.u[i * n + j] + this.u[(i + 1) * n + j]) * 0.5;
			velY = (this.v[i * n + j] + this.v[i * n + (j + 1)]) * 0.5;
		}

		let prevX;
		let prevY;

		switch (this.integrationMethod) {
			case "e":{
				// Euler
				prevX = x - (velX * dt);
				prevY = y - (velY * dt);
				break;}
			case "rk2":{
				// Rk2
				let k1x = velX*dt;
				let k1y = velY*dt;

				let midx = x-(k1x/2);
				let midy = y-(k1y/2);

				let k2x = (this.sampleField(midx, midy, "U", this.u) + this.sampleField(midx, midy, "U", this.prevU[0]))*0.5*dt;
				let k2y = (this.sampleField(midx, midy, "V", this.v) + this.sampleField(midx, midy, "V", this.prevV[0]))*0.5*dt;

				prevX = x - k2x;
				prevY = y - k2y;
				break;}
			case "rk4":{
				// Rk4
				let k1x = velX*dt;
				let k1y = velY*dt;

				let mid1x = x-(k1x/2);
				let mid1y = y-(k1y/2);

				let k2x = (this.sampleField(mid1x, mid1y, "U", this.u) + this.sampleField(mid1x, mid1y, "U", this.prevU[0]))*0.5*dt; //lerp between prev timestep and now for rk4
				let k2y = (this.sampleField(mid1x, mid1y, "V", this.v) + this.sampleField(mid1x, mid1y, "V", this.prevV[0]))*0.5*dt;
				
				let mid2x = x-(k2x/2);
				let mid2y = y-(k2y/2);

				let k3x = (this.sampleField(mid2x, mid2y, "U", this.u) + this.sampleField(mid2x, mid2y, "U", this.prevU[0]))*0.5*dt;
				let k3y = (this.sampleField(mid2x, mid2y, "V", this.v) + this.sampleField(mid2x, mid2y, "V", this.prevV[0]))*0.5*dt;

				let fullx = x-k3x;
				let fully = y-k3y;

				let k4x = this.sampleField(fullx, fully, "U", this.prevU[0])*dt;
				let k4y = this.sampleField(fullx, fully, "V", this.prevV[0])*dt;

				prevX = x - (1/6)*(k1x + 2*k2x + 2*k3x + k4x);
				prevY = y - (1/6)*(k1y + 2*k2y + 2*k3y + k4y);
			 	break;}
		}

		// Sample and store in the temporary buffer
		if (type == 'U') this.newU[i * n + j] = this.sampleField(prevX, prevY, "U", this.u);
		else if (type == 'V') this.newV[i * n + j] = this.sampleField(prevX, prevY, "V", this.v);
		else this.newM[i * n + j] = this.sampleField(prevX, prevY, "M",this.m);
	};
	
	this.sampleField = function(x, y, type, field) {
		const n = this.numY;
		
		if (type === 'U') {
			y -= 0.5;
		} else if (type === 'V') {
			x -= 0.5;
		} else { // 'M'
			x -= 0.5;
			y -= 0.5;
		}

		x = clamp(x, 0, this.numX - 2);
		y = clamp(y, 0, this.numY - 2);

		const i0 = Math.floor(x);
		const j0 = Math.floor(y);
		const i1 = i0 + 1;
		const j1 = j0 + 1;

		const tx = x - i0;
		const ty = y - j0;

		const top = lerp(field[i0 * n + j0], field[i1 * n + j0], tx);
		const bottom = lerp(field[i0 * n + j1], field[i1 * n + j1], tx);

		return lerp(top, bottom, ty);
	};

	this.getVelocityAt = function(gridX, gridY) {
		let uVal = this.sampleField(gridX, gridY, "U", this.u);
		let vVal = this.sampleField(gridX, gridY, "V", this.v);
		return [uVal, vVal];
	};

	this.drawArrows = function(offset, spacing = 4, scale) {
		const [ox, oy] = offset;
		const n = this.numY;
		
		const isSubCell = spacing < 1.0;

		for (let x = spacing; x < this.numX - spacing; x += spacing) {
			for (let y = spacing; y < this.numY - spacing; y += spacing) {
				
				let i = Math.floor(x);
				let j = Math.floor(y);
				
				if (this.s[i * n + j] === 0.0) continue; 

				let uVal, vVal;

				if (isSubCell) {
					let vel = this.getVelocityAt(x, y);
					uVal = vel[0];
					vVal = vel[1];
				} else {
					let idx = i * n + j;
					uVal = (this.u[idx] + this.u[(i + 1) * n + j]) * 0.5;
					vVal = (this.v[idx] + this.v[i * n + (j + 1)]) * 0.5;
				}

				const px = ox + x * this.h;
				const py = oy + y * this.h;

				this.drawVector(px, py, uVal * scale, vVal * scale, arrowColor);
			}
		}
	};
	
    this.draw = function(offset) {
		const [ox, oy] = offset;
		const h = this.h;
		const n = this.numY;
		
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		
		// --- PASS 1: DRAW BACKGROUNDS ---
		for (let i = 0; i < this.numX; i++) {
			for (let j = 0; j < this.numY; j++) {
				let x = Math.floor(ox + i * h);
        		let y = Math.floor(oy + j * h);
				let drawH = Math.ceil(h);

				let idx = i * n + j;
				
				if (this.showMode === "s") {
					let smoke = Math.floor(255 * (1 - this.m[idx]));
					smoke = clamp(smoke, 0, 255);
					ctx.fillStyle = `rgb(${smoke}, ${smoke}, ${smoke})`;
					if (this.s[idx] === 0.0) ctx.fillStyle = obstacleColor;
					ctx.fillRect(x, y, drawH, drawH);
				} else if (this.showMode === "p") {
					if (this.s[idx] === 0.0) {
						ctx.fillStyle = "rgb(0, 0, 0)"
						ctx.fillRect(x, y, drawH, drawH);
						continue;
					};
					
					let softIntensity = Math.tanh(this.p[idx]*this.pressureVisSensitivity); 
					let c = getSciColor(softIntensity, -1, 1);

					ctx.fillStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
					ctx.fillRect(x, y, drawH, drawH);
				} else {
					ctx.fillStyle = this.s[idx] === 0.0 ? "#222" : "#f8f8f8";
					ctx.fillRect(x, y, drawH, drawH);
					ctx.strokeStyle = "#ddd";
					ctx.lineWidth = 1;
					ctx.strokeRect(x, y, h, h);
				}
				
			}
		}
		if (this.showMode == "s" || this.showMode == "p") return;

		
		// --- PASS 2: DRAW VECTORS ---
		for (let i = 0; i < this.numX; i++) {
			for (let j = 0; j < this.numY; j++) {
				let x = ox + i * h;
				let y = oy + j * h;
				let idx = i * n + j;
				
				if (this.showMode === "f") {
					// Left Face (U) - Blue
					this.drawVector(x, y + h / 2, this.u[idx] * arrowScale, 0, "blue");
					// Top Face (V) - Green
					this.drawVector(x + h / 2, y, 0, this.v[idx] * arrowScale, "green");
					
				} else if (this.showMode === "c") {
					if (this.s[idx] > 0 && i < this.numX - 1 && j < this.numY - 1) {
						let uAvg = (this.u[idx] + this.u[(i + 1) * n + j]) * 0.5;
						let vAvg = (this.v[idx] + this.v[i * n + (j + 1)]) * 0.5;
						this.drawVector(x + h / 2, y + h / 2, uAvg * arrowScale, vAvg * arrowScale, "red");
					}
				}
			}
		}
	};
	
	this.drawVector = function(x, y, dx, dy, color) {
		const mag = Math.sqrt(dx * dx + dy * dy);
		if (mag < 0.1) return;
		
		// 1. Draw the main stem
		ctx.beginPath();
		ctx.moveTo(x, y);
		ctx.lineTo(x + dx, y + dy);
		ctx.strokeStyle = color;
		ctx.lineWidth = 2;
		ctx.stroke();
		
		const angle = Math.atan2(dy, dx);
		const headLen = 6; 
		const headAngle = Math.PI / 6; 
		
		ctx.beginPath();
		ctx.moveTo(x + dx, y + dy);
		
		ctx.lineTo(
			x + dx - headLen * Math.cos(angle - headAngle),
			y + dy - headLen * Math.sin(angle - headAngle)
		);
		
		ctx.moveTo(x + dx, y + dy);
		ctx.lineTo(
			x + dx - headLen * Math.cos(angle + headAngle),
			y + dy - headLen * Math.sin(angle + headAngle)
		);
		
		ctx.lineCap = "round";
		ctx.stroke();
	};
	
	this.setStaticCircle = function(centerX, centerY, radius) {
		let n = this.numY;
		
		for (let i = 0; i < this.numX; i++) {
			for (let j = 0; j < this.numY; j++) {
				let dx = i - centerX;
				let dy = j - centerY;
				let distanceSquared = dx * dx + dy * dy;
				
				if (distanceSquared < radius * radius) {
					let idx = i * n + j;
					this.s[idx] = 0.0;
					
					this.u[idx] = 0.0;
					this.u[(i + 1) * n + j] = 0.0;
					this.v[idx] = 0.0;
					this.v[i * n + (j + 1)] = 0.0;
				}
			}
		}
	};
	
	this.addEmitter = function(x, y, w, h, velX, velY, density = 1.0) {
		this.emitters.push({ x, y, w, h, velX, velY, density });
	};
	
	this.applyEmitters = function() {
		let n = this.numY;
		for (let e of this.emitters) {
			for (let i = e.x; i < e.x + e.w; i++) {
				for (let j = e.y; j < e.y + e.h; j++) {
					if (i < 0 || i >= this.numX || j < 0 || j >= this.numY) continue;
					
					let idx = i * n + j;
					
					this.m[idx] = e.density;
					
					this.u[idx] = e.velX;
					this.u[(i + 1) * n + j] = e.velX;
					this.v[idx] = e.velY;
					this.v[i * n + (j + 1)] = e.velY;
				}
			}
		}
	};
	
	this.setArea = function(x, y, w, h, velX, velY, den, s) {
		let n = this.numY;
		
		for (let i = x; i < x + w; i++) {
			for (let j = y; j < y + h; j++) {
				if (i < 0 || i >= this.numX || j < 0 || j >= this.numY) continue;
				
				let idx = i * n + j;
				
				this.m[idx] = den;

				this.s[idx] = s;
				
				this.u[idx] = velX;           // Left face
				this.u[(i + 1) * n + j] = velX; // Right face
				this.v[idx] = velY;           // Top face
				this.v[i * n + (j + 1)] = velY; // Bottom face
			}
		}
	};

	this.importSVG = function(svgUrl, scale = 1.0, offsetX = 0, offsetY = 0, rotation = 0) {
		const tempCanvas = document.createElement('canvas');
		tempCanvas.width = this.numX;
		tempCanvas.height = this.numY;
		const tempCtx = tempCanvas.getContext('2d');

		const img = new Image();
		img.src = svgUrl;

		img.onload = () => {
			const drawW = img.width * scale;
			const drawH = img.height * scale;

			tempCtx.save();
			// We now translate directly to the offset, treating it as the center
			tempCtx.translate(offsetX, offsetY);
			tempCtx.rotate((rotation * Math.PI) / 180);
			tempCtx.drawImage(img, -drawW / 2, -drawH / 2, drawW, drawH);
			tempCtx.restore();

			const imgData = tempCtx.getImageData(0, 0, this.numX, this.numY).data;

			for (let i = 0; i < this.numX; i++) {
				for (let j = 0; j < this.numY; j++) {
					let pixelIdx = (j * this.numX + i) * 4;
					let alpha = imgData[pixelIdx + 3];

					if (alpha > 128) {
						let n = this.numY;
						let gridIdx = i * n + j;
						this.s[gridIdx] = 0.0;
						this.m[gridIdx] = 0.0;

						this.u[gridIdx] = 0; 
						if (i + 1 < this.numX) this.u[(i + 1) * n + j] = 0; 
						this.v[gridIdx] = 0; 
						if (j + 1 < n) this.v[i * n + (j + 1)] = 0; 
					}
				}
			}
		};
	};

	this.modifyObstacle = function(mouseX, mouseY, radius, type) {
		let n = this.numY;
		let gridX = (mouseX - 50) / this.h;
		let gridY = (mouseY - 50) / this.h;

		let iStart = Math.max(1, Math.floor(gridX - radius));
		let iEnd = Math.min(this.numX - 2, Math.ceil(gridX + radius));
		let jStart = Math.max(1, Math.floor(gridY - radius));
		let jEnd = Math.min(this.numY - 2, Math.ceil(gridY + radius));

		for (let i = iStart; i <= iEnd; i++) {
			for (let j = jStart; j <= jEnd; j++) {
				let dx = (i + 0.5) - gridX;
				let dy = (j + 0.5) - gridY;
				let distSq = dx * dx + dy * dy;

				if (distSq < radius * radius) {
					const idx = i * n + j;
					
					if (type === "add") {
						this.s[idx] = 0.0;
						this.m[idx] = 0.0; // Clear smoke inside solid
						// Zero velocities on the faces of this cell
						this.u[idx] = 0;
						this.u[(i+1)*n + j] = 0;
						this.v[idx] = 0;
						this.v[i*n + j+1] = 0;
					} else {
						this.s[idx] = 1.0; // Erase (return to fluid)
					}
				}
			}
		}
	};

}

function lerp( a, b, alpha ) {
	return a + alpha * (b - a);
}

function clamp(num, lower, upper) {
	return Math.min(Math.max(num, lower), upper);
}

function sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms));
}

function getSciColor(val, minVal, maxVal) {
	val = Math.min(Math.max(val, minVal), maxVal- 0.0001);
	var d = maxVal - minVal;
	val = d == 0.0 ? 0.5 : (val - minVal) / d;
	var m = 0.25;
	var num = Math.floor(val / m);
	var s = (val - num * m) / m;
	var r, g, b;

	switch (num) {
		case 0 : r = 0.0; g = s; b = 1.0; break;
		case 1 : r = 0.0; g = 1.0; b = 1.0-s; break;
		case 2 : r = s; g = 1.0; b = 0.0; break;
		case 3 : r = 1.0; g = 1.0 - s; b = 0.0; break;
	}

	return[255*r,255*g,255*b]
}

function drawFPS() {
	ctx.fillStyle = "black";
	ctx.fillRect(5, 5, 140, 30); 
	ctx.font = "16px monospace";
	
	if (fixedFPS) {
		ctx.fillStyle = "#00FF00"; // Green for fixed
		ctx.fillText(`LOCKED: ${fps}`, 10, 25);
	} else {
		ctx.fillStyle = "#FF8800"; // Orange for free/unlocked
		ctx.fillText(`FREE: ${fps}`, 10, 25);
	}
}

function drawBrushPreview() {
	ctx.beginPath();
	ctx.arc(mouseX, mouseY, brushSize * fluid.h, 0, Math.PI * 2);
	ctx.strokeStyle = isRightDown ? "rgba(255, 0, 0, 0.5)" : "rgba(0, 255, 0, 0.5)";
	ctx.lineWidth = 2;
	ctx.stroke();
}

function drawImagePreview() {
	if (!previewImageReady || !showImagePreview) return;

	// Read slider, convert log scale to true scale
	const scaleSliderVal = parseFloat(document.getElementById("slide-img-scale")?.value || -0.3);
	const scale = Math.pow(10, scaleSliderVal);
	
	const offsetX = parseInt(document.getElementById("slide-img-x")?.value || 50);
	const offsetY = parseInt(document.getElementById("slide-img-y")?.value || 35);
	const rot = parseInt(document.getElementById("slide-img-rot")?.value || 0);

	const gridW = previewImage.width * scale;
	const gridH = previewImage.height * scale;

	const pxW = gridW * fluid.h;
	const pxH = gridH * fluid.h;

	// Center point is now exactly the offsetX and offsetY
	const centerX = 50 + offsetX * fluid.h;
	const centerY = 50 + offsetY * fluid.h;

	ctx.save();
	ctx.translate(centerX, centerY);
	ctx.rotate((rot * Math.PI) / 180);
	
	ctx.globalAlpha = 0.5;
	ctx.drawImage(previewImage, -pxW / 2, -pxH / 2, pxW, pxH);
	
	ctx.globalAlpha = 0.8;
	ctx.strokeStyle = "#00FF00";
	ctx.lineWidth = 2;
	ctx.setLineDash([5, 5]);
	ctx.strokeRect(-pxW / 2, -pxH / 2, pxW, pxH);
	
	ctx.restore();
}

function updateWalls() {
	if (!fluid) return;
	let n = fluid.numY;

	// 1. Reset all extreme edges back to fluid (s = 1.0)
	for (let i = 0; i < fluid.numX; i++) {
		fluid.s[i * n + 0] = 1.0; // Top Edge
		fluid.s[i * n + (n - 1)] = 1.0; // Bottom Edge
	}
	for (let j = 0; j < fluid.numY; j++) {
		fluid.s[0 * n + j] = 1.0; // Left Edge
		fluid.s[(fluid.numX - 1) * n + j] = 1.0; // Right Edge
	}

	// 2. Apply walls based on checkboxes
	const wTop = document.getElementById("check-wall-top")?.checked;
	const wBottom = document.getElementById("check-wall-bottom")?.checked;
	const wLeft = document.getElementById("check-wall-left")?.checked;
	const wRight = document.getElementById("check-wall-right")?.checked;

	if (wTop) fluid.setArea(0, 0, fluid.numX, 1, 0, 0, 0, 0);
	if (wBottom) fluid.setArea(0, n - 1, fluid.numX, 1, 0, 0, 0, 0);
	if (wLeft) fluid.setArea(0, 0, 1, n, 0, 0, 0, 0);
	if (wRight) fluid.setArea(fluid.numX - 1, 0, 1, n, 0, 0, 0, 0);
}

function updateEmitters() {
	if (!fluid) return;
	
	// Clear the active emitters list
	fluid.emitters = []; 

	const windEnabled = document.getElementById("check-wind")?.checked;
	const windVel = parseFloat(document.getElementById("slide-wind-vel")?.value || 100);

	const smokeEnabled = document.getElementById("check-smoke")?.checked;
	const smokeH = parseInt(document.getElementById("slide-smoke-h")?.value || 10);
	const smokeYPercent = parseFloat(document.getElementById("slide-smoke-y")?.value || 50) / 100;

	const usableY = fluid.numY - 2;

	// Add the base wind flow
	if (windEnabled) {
		fluid.addEmitter(0, 1, 2, usableY, windVel, 0, 0);
	}else{
		fluid.addEmitter(0, 1, 1, usableY, 0, 0, 0);
	}

	// Add the smoke nozzle
	if (smokeEnabled) {
		let centerY = 1 + Math.floor(usableY * smokeYPercent);
		let startY = Math.max(1, centerY - Math.floor(smokeH / 2));
		let actualH = Math.min(smokeH, fluid.numY - 1 - startY);

		fluid.addEmitter(0, startY, 2, actualH, windVel, 0, 1.0);
	}
}

function frame(currentTime) {
	requestAnimationFrame(frame);
	const elapsed = currentTime - lastFrameTime;
	
	if (!fixedFPS || elapsed >= msPerFrame) {
		
		let currentDt;

		if (fixedFPS) {
			// Adjust to maintain accurate fps pacing
			lastFrameTime = currentTime - (elapsed % msPerFrame);
			currentDt = simDt;
		} else {
			lastFrameTime = currentTime; 

			let trueDt = elapsed / 1000;

			currentDt = Math.min(trueDt, 0.1);
		}

		// --- 1. MOUSE INTERACTIONS ---
		if (isMouseDown) {
			fluid.modifyObstacle(mouseX, mouseY, brushSize, "add");
		} else if (isRightDown) {
			fluid.modifyObstacle(mouseX, mouseY, brushSize, "erase");
		}

		// --- 2. PHYSICS ---
		fluid.applyEmitters();
		fluid.makeIncompressible(incompIters);
		fluid.advect(currentDt);
		
		// --- 3. DRAWING ---
		fluid.draw([50, 50]);

		if (showArrows) {
			fluid.drawArrows([50, 50], arrowSpacing, arrowScale);
		}
		if (showPreview) {
			drawBrushPreview();
		}

		drawImagePreview();

		// --- 4. FPS COUNTER ---
		frameCount++;
		let nowInSeconds = currentTime / 1000;
		if (nowInSeconds - lastFpsUpdate >= 1.0) {
			fps = frameCount;
			frameCount = 0;
			lastFpsUpdate = nowInSeconds;
		}
		drawFPS();
	}
}

let simDt = 1/60;
let incompIters = 100;

let frameCount = 0;
let lastFpsUpdate = 0;
let fps = 0;

let lastFrameTime = 0;
let targetFps = 60;
let msPerFrame = 1000 / targetFps;

let fluid;
function initSimulation() {
	let width = parseFloat(document.getElementById("input-width").value);
	let x = parseInt(document.getElementById("input-gridx").value);
	let y = parseInt(document.getElementById("input-gridy").value);
	
	let useAspect = document.getElementById("check-aspect").checked;
	let aspectVal = parseFloat(document.getElementById("input-aspect").value);

	// 2. Calculate Y if Aspect Ratio is locked
	if (useAspect) {
		y = Math.round(x / aspectVal);
		document.getElementById("input-gridy").value = y; 
	}

	// 3. Initialize the Fluid Grid
	let cellSize = width / (x + 2);
	fluid = new Fluid(x, y, cellSize);
	
	// Resize Canvas
	canvas.width = (fluid.numX * cellSize) + 100;
	canvas.height = (fluid.numY * cellSize) + 100;

	// 4. Re-sync the physics/visual sliders so they apply to the new fluid object
	if (typeof syncUI === "function") syncUI();

	// 5. Setup Boundaries and Emitters
	updateWalls();
	updateEmitters();
}

//interaction
let isMouseDown = false;
let isRightDown = false;
let mouseX = 0;
let mouseY = 0;

let isDraggingImage = false;
let dragOffsetX = 0;
let dragOffsetY = 0;

let brushSize = 4.0; // Starting radius in grid cells
const minBrush = 0.5;
const maxBrush = 100.0;

let showPreview = true;

let showArrows = false;
let arrowSpacing = 4.0;
let arrowScale = 0.5;
let arrowColor = "#00ffff";

let obstacleColor = "#000055";

let previewImage = new Image();
let previewImageReady = false;
let showImagePreview = true;

let fixedFPS = true;

window.addEventListener("keydown", (e) => {
	if (e.key === "f" || e.key === "F") {
		fixedFPS = !fixedFPS;
		if (document.getElementById("check-fps")) {
			document.getElementById("check-fps").checked = fixedFPS;
		}
	}
});

window.addEventListener("keydown", (e) => {
	if (e.key === "v" || e.key === "V") {
		showArrows = !showArrows;
		if (document.getElementById("check-arrows")) {
			document.getElementById("check-arrows").checked = showArrows;
		}
	}
});

// Prevent the right-click menu from appearing
canvas.oncontextmenu = (e) => e.preventDefault();

canvas.addEventListener("mousedown", (e) => {
	if (e.target !== canvas) return;
	if (e.button === 1) { // Middle click prevention
		e.preventDefault();
		return; 
	} 

	updateMousePos(e);

	// --- 1. CHECK IF CLICKING THE GHOST IMAGE ---
	if (previewImageReady && showImagePreview && e.button === 0) {
		const scaleSliderVal = parseFloat(document.getElementById("slide-img-scale")?.value || -0.3);
		const scale = Math.pow(10, scaleSliderVal);

		const offsetX = parseFloat(document.getElementById("slide-img-x")?.value || 50);
		const offsetY = parseFloat(document.getElementById("slide-img-y")?.value || 35);

		// Calculate exact pixel dimensions and center
		const pxW = previewImage.width * scale * fluid.h;
		const pxH = previewImage.height * scale * fluid.h;
		const centerX = 50 + offsetX * fluid.h;
		const centerY = 50 + offsetY * fluid.h;

		// Simple rectangular hit-box detection
		if (mouseX > centerX - pxW / 2 && mouseX < centerX + pxW / 2 &&
			mouseY > centerY - pxH / 2 && mouseY < centerY + pxH / 2) {
			
			isDraggingImage = true;
			
			// Store where on the image you clicked so it doesn't snap the center to your cursor
			dragOffsetX = mouseX - centerX;
			dragOffsetY = mouseY - centerY;
			
			return; // EXIT EARLY: Do not start drawing fluid walls!
		}
	}

	// --- 2. NORMAL BRUSH DRAWING ---
	if (e.button === 0) isMouseDown = true; // Left click
	if (e.button === 2) isRightDown = true; // Right click
});

canvas.addEventListener("mousemove", (e) => {
	updateMousePos(e);

	// If we are dragging the image, update the sliders dynamically
	if (isDraggingImage) {
		// Calculate new pixel center based on mouse pos and drag offset
		const newCenterX = mouseX - dragOffsetX;
		const newCenterY = mouseY - dragOffsetY;

		// Convert pixel center back into Grid coordinates
		const newGridX = Math.round((newCenterX - 50) / fluid.h);
		const newGridY = Math.round((newCenterY - 50) / fluid.h);

		// Update HTML Sliders
		const slideX = document.getElementById("slide-img-x");
		const slideY = document.getElementById("slide-img-y");
		if (slideX) {
			slideX.value = newGridX;
			document.getElementById("val-img-x").innerText = newGridX;
		}
		if (slideY) {
			slideY.value = newGridY;
			document.getElementById("val-img-y").innerText = newGridY;
		}
	}
});

canvas.addEventListener("mouseup", (e) => {
	if (e.button === 0) {
		isMouseDown = false;
		isDraggingImage = false; // Drop the image
	}
	if (e.button === 2) isRightDown = false;
});

canvas.addEventListener("wheel", (e) => {
	if (e.target !== canvas) return;

	// e.deltaY is positive when scrolling down, negative when scrolling up
	if (e.deltaY < 0) {
		brushSize += 0.5;
	} else {
		brushSize -= 0.5;
	}

	// Clamp the value
	brushSize = Math.max(minBrush, Math.min(maxBrush, brushSize));
	
	// Update UI slider if it exists
	if (document.getElementById("slide-brush")) {
		document.getElementById("slide-brush").value = brushSize;
		document.getElementById("val-brush").innerText = brushSize.toFixed(1);
	}
	
	// Prevent the whole page from scrolling
	e.preventDefault();
}, { passive: false });

canvas.addEventListener("mousedown", (e) => {
	if (e.target !== canvas) return;
	if (e.button === 1) {
		e.preventDefault();
	}
});

canvas.addEventListener("auxclick", (e) => {
	if (e.target !== canvas) return;
	if (e.button === 1) { // Middle mouse button
		showPreview = !showPreview;
		if (document.getElementById("check-preview")) {
			document.getElementById("check-preview").checked = showPreview;
		}
	}
});

function updateMousePos(e) {
	const rect = canvas.getBoundingClientRect();
	mouseX = e.clientX - rect.left;
	mouseY = e.clientY - rect.top;
}

// --- UI CONTROLS BINDING ---
// --- UI INITIALIZATION & BINDING ---
function syncUI() {
	// 1. Simulation Options
	const slideOver = document.getElementById("slide-over");
	if (slideOver) {
		fluid.overrelaxation = parseFloat(slideOver.value);
		document.getElementById("val-over").innerText = fluid.overrelaxation.toFixed(2);
	}

	const slideDt = document.getElementById("slide-dt");
	if (slideDt) {
		simDt = parseFloat(slideDt.value);
		document.getElementById("val-dt").innerText = simDt.toFixed(3);
	}

	// 1. Simulation Physics Options
	const selectInt = document.getElementById("select-integration");
	if (selectInt) fluid.integrationMethod = selectInt.value;

	const slideIters = document.getElementById("slide-iters");
	if (slideIters) {
		incompIters = parseInt(slideIters.value);
		document.getElementById("val-iters").innerText = incompIters;
	}

	// 2. Visualisation Options
	const slidePsens = document.getElementById("slide-psens");
	if (slidePsens) {
		fluid.pressureVisSensitivity = parseFloat(slidePsens.value);
		document.getElementById("val-psens").innerText = fluid.pressureVisSensitivity.toFixed(3);
	}

	const checkFps = document.getElementById("check-fps");
	if (checkFps) fixedFPS = checkFps.checked;

	// 2. Brush Options
	const slideBrush = document.getElementById("slide-brush");
	if (slideBrush) {
		brushSize = parseFloat(slideBrush.value);
		document.getElementById("val-brush").innerText = brushSize.toFixed(1);
	}

	const checkPreview = document.getElementById("check-preview");
	if (checkPreview) showPreview = checkPreview.checked;

	// 3. Arrow Options (Logarithmic)
	const slideASpacing = document.getElementById("slide-aspacing");
	if (slideASpacing) {
		arrowSpacing = Math.pow(10, parseFloat(slideASpacing.value));
		document.getElementById("val-aspacing").innerText = arrowSpacing.toFixed(2);
	}

	const slideAScale = document.getElementById("slide-ascale");
	if (slideAScale) {
		arrowScale = Math.pow(10, parseFloat(slideAScale.value));
		document.getElementById("val-ascale").innerText = arrowScale.toFixed(2);
	}

	const colorArrow = document.getElementById("color-arrow");
	if (colorArrow) {
		arrowColor = colorArrow.value;
	}

	const slideTargetFps = document.getElementById("slide-targetfps");
	if (slideTargetFps) {
		targetFps = parseInt(slideTargetFps.value);
		msPerFrame = 1000 / targetFps;
		document.getElementById("val-targetfps").innerText = targetFps;
	}

	const checkArrows = document.getElementById("check-arrows");
	if (checkArrows) showArrows = checkArrows.checked;

	// 4. Display Mode
	const selectMode = document.getElementById("select-mode");
	if (selectMode) fluid.showMode = selectMode.value;

	const colorObs = document.getElementById("color-obstacle");
	if (colorObs) obstacleColor = colorObs.value;
}

// Attach listeners so they update during runtime
function setupListeners() {
	document.getElementById("slide-brush")?.addEventListener("input", (e) => {
		brushSize = parseFloat(e.target.value);
		document.getElementById("val-brush").innerText = brushSize.toFixed(1);
	});

	document.getElementById("slide-over")?.addEventListener("input", (e) => {
		fluid.overrelaxation = parseFloat(e.target.value);
		document.getElementById("val-over").innerText = fluid.overrelaxation.toFixed(2);
	});

	document.getElementById("slide-dt")?.addEventListener("input", (e) => {
		simDt = parseFloat(e.target.value);
		document.getElementById("val-dt").innerText = simDt.toFixed(3);
	});

	document.getElementById("slide-aspacing")?.addEventListener("input", (e) => {
		arrowSpacing = Math.pow(10, parseFloat(e.target.value));
		document.getElementById("val-aspacing").innerText = arrowSpacing.toFixed(2);
	});

	document.getElementById("slide-ascale")?.addEventListener("input", (e) => {
		arrowScale = Math.pow(10, parseFloat(e.target.value));
		document.getElementById("val-ascale").innerText = arrowScale.toFixed(2);
	});

	document.getElementById("select-integration")?.addEventListener("change", (e) => {
		fluid.integrationMethod = e.target.value;
	});

	document.getElementById("slide-iters")?.addEventListener("input", (e) => {
		incompIters = parseInt(e.target.value);
		document.getElementById("val-iters").innerText = incompIters;
	});

	document.getElementById("slide-psens")?.addEventListener("input", (e) => {
		fluid.pressureVisSensitivity = parseFloat(e.target.value);
		document.getElementById("val-psens").innerText = fluid.pressureVisSensitivity.toFixed(3);
	});

	document.getElementById("color-arrow")?.addEventListener("input", (e) => {
		arrowColor = e.target.value;
	});

	document.getElementById("color-obstacle")?.addEventListener("input", (e) => {
		obstacleColor = e.target.value;
	});

	document.getElementById("slide-targetfps")?.addEventListener("input", (e) => {
		targetFps = parseInt(e.target.value);
		msPerFrame = 1000 / targetFps; // Recalculate timing threshold
		document.getElementById("val-targetfps").innerText = targetFps;
	});

	document.getElementById("check-fps")?.addEventListener("change", (e) => fixedFPS = e.target.checked);
	document.getElementById("check-arrows")?.addEventListener("change", (e) => showArrows = e.target.checked);
	document.getElementById("check-preview")?.addEventListener("change", (e) => showPreview = e.target.checked);
	document.getElementById("select-mode")?.addEventListener("change", (e) => fluid.showMode = e.target.value);

	// --- Grid Setup Listeners ---
	const checkAspect = document.getElementById("check-aspect");
	const inputAspect = document.getElementById("input-aspect");
	const inputGridY = document.getElementById("input-gridy");

	if (checkAspect && inputGridY && inputAspect) {
		checkAspect.addEventListener("change", (e) => {
			if (e.target.checked) {
				inputGridY.disabled = true;
				inputAspect.disabled = false;
				// Immediately calculate and update Y
				let xVal = parseInt(document.getElementById("input-gridx").value);
				let aspectVal = parseFloat(inputAspect.value);
				inputGridY.value = Math.round(xVal / aspectVal);
			} else {
				inputGridY.disabled = false;
				inputAspect.disabled = true;
			}
		});
	}

	const btnRestart = document.getElementById("btn-restart");
	if (btnRestart) {
		btnRestart.addEventListener("click", () => {
			initSimulation();
		});
	}

	// --- Image Import Listeners ---
	let currentImageUrl = null;
	
	const inputFile = document.getElementById("input-file");
	if (inputFile) {
		inputFile.addEventListener("change", (e) => {
			const file = e.target.files[0];
			if (file) {
				if (currentImageUrl) URL.revokeObjectURL(currentImageUrl);
				currentImageUrl = URL.createObjectURL(file);
				
				// Load it into our preview object
				previewImage.src = currentImageUrl;
				previewImage.onload = () => {
					previewImageReady = true;
				};
			}
		});
	}

	const checkImgPreview = document.getElementById("check-img-preview");
	if (checkImgPreview) {
		checkImgPreview.addEventListener("change", (e) => {
			showImagePreview = e.target.checked;
		});
	}

	document.getElementById("slide-img-scale")?.addEventListener("input", (e) => {
		const trueScale = Math.pow(10, parseFloat(e.target.value));
		document.getElementById("val-img-scale").innerText = trueScale.toFixed(2);
	});

	document.getElementById("slide-img-x")?.addEventListener("input", (e) => {
		document.getElementById("val-img-x").innerText = e.target.value;
	});

	document.getElementById("slide-img-y")?.addEventListener("input", (e) => {
		document.getElementById("val-img-y").innerText = e.target.value;
	});

	document.getElementById("slide-img-rot")?.addEventListener("input", (e) => {
		document.getElementById("val-img-rot").innerText = e.target.value;
	});

	const btnImport = document.getElementById("btn-import");
	if (btnImport) {
		btnImport.addEventListener("click", () => {
			if (!currentImageUrl) {
				alert("Please select an image file first.");
				return;
			}
			
			// Convert log value to true scale before passing to the fluid engine
			const logScale = parseFloat(document.getElementById("slide-img-scale").value);
			const scale = Math.pow(10, logScale);
			
			const offsetX = parseInt(document.getElementById("slide-img-x").value);
			const offsetY = parseInt(document.getElementById("slide-img-y").value);
			const rot = parseInt(document.getElementById("slide-img-rot").value);

			fluid.importSVG(currentImageUrl, scale, offsetX, offsetY, rot);
		});
	}

	// --- Environment Listeners ---
	const envInputs = ["check-wall-top", "check-wall-bottom", "check-wall-left", "check-wall-right"];
	envInputs.forEach(id => {
		document.getElementById(id)?.addEventListener("change", updateWalls);
	});

	document.getElementById("check-wind")?.addEventListener("change", updateEmitters);
	document.getElementById("check-smoke")?.addEventListener("change", updateEmitters);

	document.getElementById("slide-wind-vel")?.addEventListener("input", (e) => {
		document.getElementById("val-wind-vel").innerText = e.target.value;
		updateEmitters();
	});

	document.getElementById("slide-smoke-h")?.addEventListener("input", (e) => {
		document.getElementById("val-smoke-h").innerText = e.target.value;
		updateEmitters();
	});

	document.getElementById("slide-smoke-y")?.addEventListener("input", (e) => {
		document.getElementById("val-smoke-y").innerText = e.target.value;
		updateEmitters();
	});

	// Collapsible Logic
	const headers = document.getElementsByClassName("collapsible-header");
	for (let i = 0; i < headers.length; i++) {
		headers[i].addEventListener("click", function() {
			this.parentElement.classList.toggle("active");
		});
	}
}

setupListeners();
initSimulation();
requestAnimationFrame(frame);