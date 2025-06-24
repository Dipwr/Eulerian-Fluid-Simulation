

function Fluid(){
    this.gridDim = [1000,1000];
    this.cellSize = 10;

    this.x = this.gridDim[0]/this.cellSize;
    this.y = this.gridDim[1]/this.cellSize;
    
    this.overrelaxationCoef = 1.5;

    this.points = []
    

    this.setup = function(){
        for (let i = 0; i < (this.x*2)+1; i++){this.points.push([]);}

        for (let i = 0; i < this.x+1; i++){
            for (let j = 0; j < this.y; j++){
                if ((j <= 1 || j >= this.y-2) || (i == 0 || i == this.x)){
                    this.points[i*2][j] = new Point([this.cellSize*i, this.cellSize*(0.5+j)], 0, 0, true); //left point
                }else {
                    this.points[i*2][j] = new Point([this.cellSize*i, this.cellSize*(0.5+j)], 0, 0, false); //left point
                }
            }
        }
        for (let i = 0; i < this.x; i++){
            for (let j = 0; j < this.y+1; j++){
                if ((j <= 1 || j >= this.y-1) || (i == 0 || i == this.x-1)){
                    this.points[(i*2)+1][j] = new Point([this.cellSize*(0.5+i), this.cellSize*(j)], 0, 0, true); //left point
                }else{
                    this.points[(i*2)+1][j] = new Point([this.cellSize*(0.5+i), this.cellSize*(j)], 0, 0, false); //left point
                }
            }
        }

    }

    this.makeIncompressible = function(iterations){
        for (let iteration = 0; iteration < iterations; iteration++){
            for (let i = 0; i < this.x; i++){
                for (let j = 0; j < this.y; j++){
                    let cellPoints = this.cellToPoints(i,j);

                    let acc = 0
                    let advectingPoints = [];
                    for (let k = 0; k < 4; k++){
                        let PC = cellPoints[k];//point coords
                        let point = this.points[PC[0]][PC[1]];

                        acc += point.vel * (((k==1) || (k==2))? -1 : 1);

                        if (!point.skipAdvection){advectingPoints.push(k);}
                    }
                    for (let l = 0; l < advectingPoints.length; l++){
                        let k = advectingPoints[l];
                        let PC = cellPoints[k];//point coords
                        let point = this.points[PC[0]][PC[1]];

                        point.vel += (acc/advectingPoints.length)* this.overrelaxationCoef * (((k==1) || (k==2))? 1 : -1);
                    }
                }
            }
        }
    }

    this.advect = function(dt){
        for (let i = 2; i < (this.x*2)-1; i++){
            for (let j = 2; j < this.points[i].length-2; j++){
                let point = []

                // --- advect velocities ---

                if (i % 2 == 0){
                    //horizontal velocities
                    let vel = []
                    vel[0] = this.points[i][j].vel;

                    let verticalVelAcc = this.points[i-1][j].vel + this.points[i-1][j+1].vel + this.points[i+1][j].vel + this.points[i+1][j+1].vel;
                    vel[1] = verticalVelAcc/4;

                    point[0] = this.points[i][j].pos[0] - (vel[0] * dt);
                    point[1] = this.points[i][j].pos[1] - (vel[1] * dt);

                    let pX = Math.floor(point[0]/this.cellSize)*2;
                    let pY = Math.floor((point[1]/this.cellSize)+0.5);

                    let interpolation1 = lerp(this.points[pX][pY-1].vel, this.points[pX][pY].vel, ((point[1]/this.cellSize) - 0.5) % 1);
                    let interpolation2 = lerp(this.points[pX+2][pY-1].vel, this.points[pX+2][pY].vel, ((point[1]/this.cellSize) - 0.5) % 1);
                    let interpolation3 = lerp(interpolation1, interpolation2, (point[0]/this.cellSize) % 1);

                    this.points[i][j].vel = interpolation3;
                }else{
                    //vertical velocities
                    let vel = []
                    vel[1] = this.points[i][j].vel;

                    let verticalVelAcc = this.points[i-1][j-1].vel + this.points[i-1][j].vel + this.points[i+1][j-1].vel + this.points[i+1][j].vel;
                    vel[0] = verticalVelAcc/4;

                    point[0] = this.points[i][j].pos[0] - (vel[0] * dt);
                    point[1] = this.points[i][j].pos[1] - (vel[1] * dt);

                    let pX = (Math.floor(point[0]/this.cellSize)*2) + 1;
                    let pY = Math.ceil(point[1]/this.cellSize);

                    let interpolation1 = lerp(this.points[pX][pY-1].vel, this.points[pX][pY].vel, (point[1]/this.cellSize) % 1);
                    let interpolation2 = lerp(this.points[pX+2][pY-1].vel, this.points[pX+2][pY].vel, (point[1]/this.cellSize) % 1);
                    let interpolation3 = lerp(interpolation1, interpolation2, ((point[0]/this.cellSize) - 0.5) % 1);

                    this.points[i][j].vel = interpolation3;
                }
            }
        }
    }

    this.cellToPoints = function(x,y){
        return [[2*x, y],[(2*x)+1, y],[2*(x+1), y],[(2*x)+1, y+1]]//left, up, right, down
    }
}

function Point(pos, vel, smokeDen, skipAdvection){
    this.pos = pos;
    this.vel = vel;
    this.smokeDen = smokeDen;

    this.skipAdvection = skipAdvection;
}

function lerp( a, b, alpha ) {
    return a + alpha * (b - a);
}

let fluid = new Fluid();
fluid.setup();
fluid.points[0][3].vel = 5;
fluid.makeIncompressible(2);
fluid.advect(1);
fluid.makeIncompressible(2);
fluid.advect(1);
fluid.makeIncompressible(2);
fluid.advect(1);
fluid.makeIncompressible(2);
fluid.advect(1);
fluid.makeIncompressible(2);
fluid.advect(1);
fluid.makeIncompressible(2);
fluid.advect(1);
fluid.makeIncompressible(2);
fluid.advect(1);
fluid.makeIncompressible(2);
fluid.advect(1);
fluid.makeIncompressible(2);